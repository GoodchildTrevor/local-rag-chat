from __future__ import annotations

import asyncio
import json
import os
import re
import psutil
import shutil
import time
import tempfile
from datetime import datetime, timezone
from logging import Logger
from pathlib import Path
from typing import Any
from functools import lru_cache
from collections import deque

import aiohttp
import fitz
from docx2python import docx2python
from razdel import sentenize, tokenize

from config.settings import AppConfig, NLPConfig
from databases.document_upserting.processing_excel import (
    excel_to_text,
    extract_excel_metadata,
)
from databases.document_upserting.processing_utils import (
    normalize_datetime,
    safe_decode,
)
from databases.document_upserting.processing_word import (
    convert_doc_to_docx,
    word_to_text
)
from databases.document_upserting.processing_pdf import (
    iter_pdf_text_batches
)


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def preprocess_text(
    logger: Logger,
    nlp_config: NLPConfig,
    text: str,
) -> list[dict[str, Any]]:
    """
    Tokenize and lemmatize raw text into sentence-level units.

    Each returned sentence dict contains:
      - raw: original sentence text
      - lemmas: space-joined lemma tokens (stopwords filtered)
      - pairs: list of (raw_token, lemma) pairs, used by split_long_sentence
    """
    sentences = list(sentenize(text))
    processed: list[dict[str, Any]] = []

    for s in sentences:
        raw = s.text.strip()
        if not raw:
            continue

        tokens: list[str] = [
            t.text.lower()
            for t in tokenize(raw)
            if any(c.isalnum() for c in t.text)
            and (
                t.text.lower() not in nlp_config.stopwords
                or t.text.isupper()
            )
        ]

        if not tokens:
            logger.debug(
                "Skipping empty or non-alphabetic sentence: '%s...'", raw[:30]
            )
            continue

        lemmas: list[str] = []
        pairs: list[tuple[str, str]] = []

        for tok in tokens:
            parsed = nlp_config.morph.parse(tok)
            lemma = parsed[0].normal_form if parsed else tok
            lemmas.append(lemma)
            pairs.append((tok, lemma))

        lemmatized = " ".join(lemmas).strip()
        if not lemmatized:
            logger.debug(
                "Skipping sentence with empty lemma: '%s...'", raw[:30]
            )
            continue

        processed.append(
            {
                "raw": raw,
                "lemmas": lemmatized,
                "pairs": pairs,
            }
        )

    logger.info(
        "Tokenization and lemmatization finished: %d valid sentences",
        len(processed),
    )
    return processed


def split_long_sentence(
    logger: Logger,
    nlp_config: NLPConfig,
    sentence: dict[str, Any],
    max_tokens: int,
) -> list[dict[str, str]]:
    """
    Iteratively split an oversized sentence into smaller parts.

    Uses the ``pairs`` list from preprocess_text as the source of truth
    so ``raw`` and ``lemmas`` in every part stay aligned.

    A queue (FIFO) is used to preserve left‑to‑right order in the final
    ``parts`` list.
    """
    tokenizer = nlp_config.tokenizer
    token_cache: dict[str, int] = {}

    def count(text: str) -> int:
        if text in token_cache:
            return token_cache[text]
        val = len(tokenizer.encode(text, disallowed_special=()))
        token_cache[text] = val
        return val

    pairs: list[tuple[str, str]] = sentence.get("pairs", [])
    raw_text = sentence.get("raw", "").strip()
    lemmas_text = sentence.get("lemmas", "").strip()

    if not pairs or not raw_text:
        return [{"raw": raw_text, "lemmas": lemmas_text}]

    threshold = int(max_tokens * 1.5)

    if count(lemmas_text) <= threshold:
        return [{"raw": raw_text, "lemmas": lemmas_text}]

    # Queue entries are (start, end) index ranges into `pairs`.
    queue = deque()
    queue.append((0, len(pairs)))
    parts: list[dict[str, str]] = []

    while queue:
        start, end = queue.popleft()
        slice_pairs = pairs[start:end]
        part_raw = " ".join(r for r, _ in slice_pairs)
        part_lemmas = " ".join(l for _, l in slice_pairs)

        if count(part_lemmas) <= threshold:
            parts.append({"raw": part_raw, "lemmas": part_lemmas})
            continue

        # Degenerate single‑token slice: cannot be split further.
        if end - start <= 1:
            parts.append({"raw": part_raw, "lemmas": part_lemmas})
            continue

        mid = (start + end) // 2
        queue.append((start, mid))
        queue.append((mid, end))

    if not parts:
        logger.warning(
            "Splitting produced no parts, returning original sentence: '%s...'",
            raw_text[:50],
        )
        return [{"raw": raw_text, "lemmas": lemmas_text}]

    return parts


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def text_chunker(
    logger: Logger,
    nlp_config: NLPConfig,
    text: str,
    max_tokens: int,
    overlap: int,
    min_tokens: int = 3,
) -> list[dict[str, Any]]:
    """
    Split text into chunks respecting sentence boundaries.

    Упрощённая и надёжная схема:
    - Один индекс i идёт слева направо по предложениям.
    - Для overlap используем стартовый индекс max(0, i - overlap).
    - Никаких вычислений chunk_start через _meta предыдущего чанка.
    - Длинные предложения (> max_tokens) обрабатываются отдельно через split_long_sentence.
    """

    processed = preprocess_text(logger, nlp_config, text)
    if not processed:
        return []

    # Предварительно считаем токены для всех предложений батчами
    lemma_texts = [s["lemmas"] for s in processed]
    token_counts: list[int] = [0] * len(lemma_texts)

    batch_size = 256
    logger.info(
        "Chunking %d sentences (max_tokens=%d, overlap=%d)",
        len(lemma_texts), max_tokens, overlap,
    )

    for start in range(0, len(lemma_texts), batch_size):
        end = start + batch_size
        batch = lemma_texts[start:end]
        encoded_batch = nlp_config.tokenizer.encode_batch(
            batch,
            disallowed_special=(),
        )
        for local_idx, encoded in enumerate(encoded_batch):
            global_idx = start + local_idx
            token_counts[global_idx] = len(encoded)

    sentences: list[dict[str, Any]] = []
    for idx, s in enumerate(processed):
        sentences.append(
            {
                "raw": s["raw"],
                "lemmas": s["lemmas"],
                "tokens": token_counts[idx],
                "pairs": s.get("pairs", []),
            }
        )

    chunks: list[dict[str, Any]] = []
    total = len(sentences)
    i = 0

    while i < total:
        # базовый старт чанка с overlap по предложениям
        start_idx = max(0, i - overlap)

        current_raw: list[str] = []
        current_lemmas: list[str] = []
        current_tokens = 0

        j = start_idx
        while j < total:
            sent = sentences[j]
            sent_tokens = sent["tokens"]

            # Слишком длинное предложение: обрабатываем отдельно
            if sent_tokens > max_tokens:
                logger.debug(
                    "Sentence %d too large (%d tokens > %d), splitting...",
                    j, sent_tokens, max_tokens,
                )

                # если что-то уже накопили в текущем чанке — сначала его зафиксируем
                if current_raw and current_tokens >= min_tokens:
                    chunks.append(
                        {
                            "raw": " ".join(current_raw),
                            "lemmas": " ".join(current_lemmas),
                            "_meta": {
                                "tokens": current_tokens,
                                "sentences": j - start_idx,
                                "start_sentence": start_idx,
                                "end_sentence": j - 1,
                            },
                        }
                    )
                    current_raw = []
                    current_lemmas = []
                    current_tokens = 0

                # сплитим одно предложение в несколько частей
                long_parts = split_long_sentence(
                    logger=logger,
                    nlp_config=nlp_config,
                    sentence=sent,
                    max_tokens=max_tokens,
                )
                if long_parts:
                    lemma_parts = [p["lemmas"] for p in long_parts]
                    encoded_parts = nlp_config.tokenizer.encode_batch(
                        lemma_parts,
                        disallowed_special=(),
                    )
                    for part_idx, (part, encoded) in enumerate(
                        zip(long_parts, encoded_parts), start=1
                    ):
                        part_tokens = len(encoded)
                        if part_tokens < min_tokens:
                            logger.debug(
                                "Skipping too short split part (%d tokens) from sentence %d",
                                part_tokens,
                                j,
                            )
                            continue
                        chunks.append(
                            {
                                "raw": part["raw"],
                                "lemmas": part["lemmas"],
                                "_meta": {
                                    "tokens": part_tokens,
                                    "sentences": 1,
                                    "start_sentence": j,
                                    "end_sentence": j,
                                    "from_long_sentence": True,
                                    "part": part_idx,
                                },
                            }
                        )

                # переходим к следующему предложению
                j += 1
                # базовый индекс i двигается вперёд минимум на 1
                i = max(i + 1, j)
                break

            # если добавление этого предложения переполнит чанк — фиксируем его
            if current_tokens + sent_tokens > max_tokens:
                break

            current_raw.append(sent["raw"])
            current_lemmas.append(sent["lemmas"])
            current_tokens += sent_tokens
            j += 1

        # сохраним нормальный чанк, если он не пустой
        if current_raw and current_tokens >= min_tokens:
            chunks.append(
                {
                    "raw": " ".join(current_raw),
                    "lemmas": " ".join(current_lemmas),
                    "_meta": {
                        "tokens": current_tokens,
                        "sentences": j - start_idx,
                        "start_sentence": start_idx,
                        "end_sentence": j - 1,
                    },
                }
            )

        # двигаем i вперёд: либо до конца окна, либо минимум на 1
        if j <= i:
            # страховка от застревания
            logger.warning(
                "Cursor stuck at %d (j=%d), forcing advance by 1 sentence.",
                i,
                j,
            )
            i += 1
        else:
            i = j

        if i % 100 == 0 or i >= total:
            logger.info(
                "Progress: %d/%d sentences, %d chunks",
                i,
                total,
                len(chunks),
            )

    avg_tokens = (
        sum(ch["_meta"].get("tokens", 0) for ch in chunks) / max(1, len(chunks))
    )
    logger.info(
        "Chunking finished: %d chunks, avg tokens: %.1f",
        len(chunks),
        avg_tokens,
    )
    return chunks


def chunker(
    logger: Logger,
    nlp_config: NLPConfig,
    elements: list[dict[str, Any]],
    max_tokens: int,
    overlap: int,
    min_tokens: int = 3,
) -> list[dict[str, Any]]:
    """
    Process and chunk document elements based on their type.

    Supported types:
      - text: chunked with text_chunker
      - table: each row becomes a separate chunk; tokenizer calls are batched
      - image: whole description becomes a chunk
    """
    chunks: list[dict[str, Any]] = []

    for el in elements:
        element_type = el.get("type")
        content = el.get("content")
        base_meta = dict(el.get("_meta", {}))

        try:
            if element_type == "text":
                if not content:
                    continue
                text_content = content[0]
                if not text_content or not text_content.strip():
                    continue

                text_chunks = text_chunker(
                    logger=logger,
                    nlp_config=nlp_config,
                    text=text_content,
                    max_tokens=max_tokens,
                    overlap=overlap,
                    min_tokens=min_tokens,
                )
                for chunk in text_chunks:
                    meta = dict(base_meta)
                    meta.update(chunk.get("_meta", {}))
                    chunk["_meta"] = meta
                chunks.extend(text_chunks)
                logger.info(
                    "Processed text element into %d chunks", len(text_chunks)
                )

            elif element_type == "table":
                table_marker = base_meta.get("table_marker", "unknown")
                page_meta = {
                    k: v
                    for k, v in base_meta.items()
                    if k in ("page_start", "page_end")
                }

                # Step 1: preprocess all non-empty rows (morph.parse happens here).
                row_data: list[tuple[int, str, str]] = []
                for row_idx, row in enumerate(content):
                    if not row or not row.strip():
                        continue
                    processed_rows = preprocess_text(logger, nlp_config, row)
                    if not processed_rows:
                        continue
                    combined_raw = " ".join(p["raw"] for p in processed_rows)
                    combined_lemmas = " ".join(p["lemmas"] for p in processed_rows)
                    row_data.append((row_idx, combined_raw, combined_lemmas))

                # Step 2: batch-encode all lemmas in a single call.
                if row_data:
                    lemma_texts = [lemmas for _, _, lemmas in row_data]
                    encoded_batch = nlp_config.tokenizer.encode_batch(
                        lemma_texts, disallowed_special=()
                    )
                    for (row_idx, combined_raw, combined_lemmas), encoded in zip(
                        row_data, encoded_batch
                    ):
                        token_count = len(encoded)
                        if token_count < min_tokens:
                            logger.debug(
                                "Table row too short (%d tokens), skipping",
                                token_count,
                            )
                            continue
                        chunks.append(
                            {
                                "raw": combined_raw,
                                "lemmas": combined_lemmas,
                                "_meta": {
                                    "table_row": True,
                                    "row_index": row_idx,
                                    "table_marker": table_marker,
                                    **page_meta,
                                },
                            }
                        )

                logger.info("Processed table with %d rows", len(content))

            elif element_type == "image":
                if not content:
                    continue
                image_text = content[0]
                if not image_text or not image_text.strip():
                    continue

                processed_image = preprocess_text(logger, nlp_config, image_text)
                if processed_image:
                    combined_raw = " ".join(p["raw"] for p in processed_image)
                    combined_lemmas = " ".join(p["lemmas"] for p in processed_image)
                    token_count = len(
                        nlp_config.tokenizer.encode(
                            combined_lemmas, disallowed_special=()
                        )
                    )
                    if token_count < min_tokens:
                        logger.debug(
                            "Image description too short (%d tokens), skipping",
                            token_count,
                        )
                        continue

                    chunks.append(
                        {
                            "raw": combined_raw,
                            "lemmas": combined_lemmas,
                            "_meta": dict(base_meta),
                        }
                    )
                    logger.info("Processed image element")

            else:
                logger.warning("Unknown element type: %s, skipping", element_type)

        except Exception as e:
            logger.error("Error %s during processing %s", e, element_type)
            raise

    filtered_chunks = [
        chunk for chunk in chunks
        if any(c.isalnum() for c in chunk.get("raw", ""))
        and not re.search(r"\b(nan|inf)\b", chunk.get("raw", ""), re.IGNORECASE)
    ]

    logger.info(
        "Total chunks created: %d (filtered out %d)",
        len(filtered_chunks),
        len(chunks) - len(filtered_chunks),
    )
    return filtered_chunks


# ---------------------------------------------------------------------------
# Table extraction from text
# ---------------------------------------------------------------------------

def detect_and_extract_tables(logger: Logger, text: str) -> dict[str, Any]:
    """
    Detect and extract tables from text returned by external service.

    Tables are expected to be JSON-like arrays of rows inside square brackets.
    To reduce false positives, only brackets whose content starts with a quote
    or bracket are considered candidates.
    """
    table_pattern = r"\[(.*?)\]"
    tables: list[list[str]] = []
    table_map: dict[str, list[str]] = {}

    def replacer(match: re.Match[str]) -> str:
        nonlocal tables
        content = match.group(1).strip()
        if not content:
            return match.group(0)

        # Heuristic: only try to parse as table if it looks like a list of strings.
        if not (content.startswith("'") or content.startswith('"')):
            return match.group(0)

        try:
            data = json.loads(f"[{content}]")
            if isinstance(data, list) and all(isinstance(row, str) for row in data):
                marker = f"[TABLE_{len(tables) + 1}]"
                tables.append(data)
                table_map[marker] = data
                return marker
        except json.JSONDecodeError:
            logger.debug(
                "JSON parse failed for table candidate, falling back to naive split: '[%s...]'",
                content[:50],
            )

        rows: list[str] = []
        for part in content.split(","):
            part = part.strip()
            if not part:
                continue
            if (part.startswith("'") and part.endswith("'")) or (
                part.startswith('"') and part.endswith('"')
            ):
                part = part[1:-1]
            if part:
                rows.append(part)

        if rows:
            marker = f"[TABLE_{len(tables) + 1}]"
            tables.append(rows)
            table_map[marker] = rows
            return marker

        return match.group(0)

    cleaned_text = re.sub(table_pattern, replacer, text, flags=re.DOTALL)

    return {
        "cleaned_text": cleaned_text,
        "tables": tables,
        "table_map": table_map,
    }


# ---------------------------------------------------------------------------
# Unified extraction
# ---------------------------------------------------------------------------

async def extract_text_metadata(
    logger: Logger,
    app_config: AppConfig,
    file_path: Path,
    file_format: str,
    nlp_config: NLPConfig,
    session: aiohttp.ClientSession,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Extract textual content and metadata from a given file.
    """
    elements: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}

    # ------------------------------------------------------------------ PDF
    if file_format == ".pdf":
        try:
            with fitz.open(file_path) as doc:
                metadata["creation_date"] = normalize_datetime(
                    doc.metadata.get("creationDate", "")
                )
                metadata["modification_date"] = normalize_datetime(
                    doc.metadata.get("modDate", "")
                )
        except Exception as e:
            logger.warning("Failed to extract PDF metadata: %s", e)
            fallback = datetime(1900, 1, 1, tzinfo=timezone.utc)
            metadata["creation_date"] = fallback
            metadata["modification_date"] = fallback

        try:
            async for part in iter_pdf_text_batches(
                logger, app_config, file_path, session, page_batch_size=1
            ):
                page_start = part["page_start"]
                page_end = part["page_end"]
                text = part["text"]

                result = detect_and_extract_tables(logger, text)
                cleaned_text = result["cleaned_text"]
                table_map = result["table_map"]

                if cleaned_text.strip():
                    elements.append(
                        {
                            "type": "text",
                            "content": [cleaned_text],
                            "_meta": {
                                "page_start": page_start,
                                "page_end": page_end,
                            },
                        }
                    )

                for marker, table_rows in table_map.items():
                    if not table_rows:
                        continue
                    elements.append(
                        {
                            "type": "table",
                            "content": table_rows,
                            "_meta": {
                                "table_marker": marker,
                                "page_start": page_start,
                                "page_end": page_end,
                            },
                        }
                    )
                    logger.info(
                        "Extracted %s from PDF pages %d-%d",
                        marker,
                        page_start,
                        page_end,
                    )

        except Exception as e:
            logger.error("PDF text extraction failed: %s", e)
            raise RuntimeError(f"PDF processing failed: {e}") from e

    # ------------------------------------------------------------ DOC / DOCX
    elif file_format in (".docx", ".doc"):
        current_path = file_path

        if file_format == ".doc":
            try:
                converted_path = await convert_doc_to_docx(
                    file_path, logger, app_config.libreoffice_timeout
                )
                if converted_path is None:
                    raise RuntimeError("DOC to DOCX conversion failed")
                current_path = Path(converted_path)
            except Exception as e:
                logger.error("Failed to convert .doc to .docx: %s", e)
                raise RuntimeError(f"DOC conversion failed: {e}") from e

        try:
            with docx2python(current_path) as doc_result:
                raw_metadata = {
                    k: safe_decode(v)
                    for k, v in doc_result.core_properties.items()
                }
                metadata["creation_date"] = normalize_datetime(
                    raw_metadata.get("created", "")
                )
                metadata["modification_date"] = normalize_datetime(
                    raw_metadata.get("modified", "")
                )
        except Exception as e:
            logger.warning("Failed to extract DOCX metadata: %s", e)
            fallback = datetime(1900, 1, 1, tzinfo=timezone.utc)
            metadata["creation_date"] = fallback
            metadata["modification_date"] = fallback

        try:
            elements = await word_to_text(
                logger, app_config, current_path, session
            )
        except Exception as e:
            logger.error("Failed to extract text from Word document: %s", e)
            raise RuntimeError(f"Word text extraction failed: {e}") from e

    # ------------------------------------------------------------------ DJVU
    elif file_format == ".djvu":
        logger.warning("DJVU format is not implemented yet: %s", file_path)
        return [], {}

    # ----------------------------------------------------------------- XLSX
    elif file_format == ".xlsx":
        elements = excel_to_text(file_path, logger, nlp_config)
        metadata = extract_excel_metadata(file_path, logger)

    else:
        error_msg = f"Unsupported file format: {file_format}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    return elements, metadata
