from collections.abc import Iterable
from pathlib import Path

from datetime import datetime, timezone, timedelta
from logging import Logger
import os
import re

import aspose.words as aw
from docx2python import docx2python
import fitz
from fitz import Document

import pytesseract
from PIL import Image
from razdel import sentenize, tokenize

from config.consts.database import (
    PDF_SIZE_LIMIT,
    DPI,
)
from config.settings import NLPConfig


def preprocess_text(logger: Logger, nlp_config: NLPConfig, text: str) -> list[dict]:
    """
    Tokenize and lemmatize raw text into sentence-level units.
    :param logger: Logger instance for tracking pipeline execution.
    :param nlp_config: NLP configuration object containing tokenizer, stopwords and morphological analyzer.
    :param text: Input raw text.
    :return: A list of dictionaries, each containing raw and lemmatized text of a sentence.
    """
    sentences = list(sentenize(text))
    processed = []

    for s in sentences:
        raw = s.text.strip()
        tokens = [
            t.text.lower() for t in tokenize(raw)
            if t.text.isalpha() and t.text.lower() not in nlp_config.stopwords and len(t.text) > 1
        ]
        if not tokens:
            logger.debug(f"Skipping empty or non-alphabetic sentence: '{raw[:30]}...'")
            continue

        lemmas = [nlp_config.morph.parse(tok)[0].normal_form for tok in tokens]
        lemmatized = " ".join(lemmas)
        if not lemmatized.strip():
            logger.debug(f"Skipping sentence with empty lemma: '{raw[:30]}...'")
            continue

        processed.append({"raw": raw, "lemmas": lemmatized})

    logger.info(f"Tokenization and lemmatization finished: {len(processed)} valid sentences")
    return processed


def chunker(
        logger: Logger, 
        nlp_config: NLPConfig, 
        text: str, 
        max_tokens: int, 
        overlap: int
) -> list[dict]:
    """
    Split text into chunks constrained by maximum token length.
    Each chunk is created based on the lemmatized representation of sentences.
    Overlap between chunks is added by including a configurable number of
    tokens from the previous chunk.
    :param logger: Logger instance for tracking pipeline execution.
    :param nlp_config: NLP configuration object containing tokenizer, stopwords and morphological analyzer.
    :param text: Input raw text.
    :param max_tokens: Maximum number of tokens allowed per chunk.
    :param overlap: Number of overlapping sentences (converted to tokens) to include from the previous chunk.
    :return: A list of dictionaries, each containing raw and lemmatized text of a chunk.
    """
    processed = preprocess_text(logger, nlp_config, text)
    chunks = []
    i = 0

    while i < len(processed):
        current_chunk_raw = []
        current_chunk_lemmas = []
        current_token_count = 0

        # Add overlap from the previous chunk
        if chunks and overlap > 0:
            last_chunk_lemmas = chunks[-1]["lemmas"].split()
            overlap_lemmas = last_chunk_lemmas[-overlap * 2:]
            overlap_raw = chunks[-1]["raw"].split()[-overlap * 2:]
            current_chunk_lemmas.extend(overlap_lemmas)
            current_chunk_raw.extend(overlap_raw)
            current_token_count = sum(
                len(nlp_config.tokenizer.encode(lemma, disallowed_special=()))
                for lemma in overlap_lemmas
            )
            logger.debug(f"Added overlap: {len(overlap_lemmas)} tokens")

        print(i)

        # Build chunk
        while i < len(processed) and current_token_count <= max_tokens:
            sent = processed[i]
            lemma_sent = sent["lemmas"]
            token_count = len(nlp_config.tokenizer.encode(lemma_sent, disallowed_special=()))

            if current_token_count + token_count > max_tokens:
                if not current_chunk_raw:
                    # Sentence is too long -> save separately
                    chunks.append({
                        "raw": sent["raw"],
                        "lemmas": lemma_sent,
                    })
                    logger.warning(
                        f"⚠️ Sentence {i} is too long "
                        f"({token_count} tokens), stored separately"
                    )
                    i += 1
                else:
                    logger.debug(
                        f"Chunk reached limit ({current_token_count} tokens), finishing chunk"
                    )
                pass

            current_chunk_raw.append(sent["raw"])
            current_chunk_lemmas.append(lemma_sent)
            current_token_count += token_count
            i += 1

        if current_chunk_raw:
            chunks.append({
                "raw": " ".join(current_chunk_raw),
                "lemmas": " ".join(current_chunk_lemmas),
            })
            logger.debug(
                f"✅ Chunk created: {current_token_count} tokens, "
                f"{len(current_chunk_raw)} sentences"
            )

    return chunks


def format_date(date_str: str) -> datetime:
    """
    Processing different types of datetime for metadata information
    :param date_str: raw date in string format
    :return: correct datetime
    """
    default_date = datetime(1900, 1, 1, 0, 0, 0)
    if not date_str or not date_str.startswith("D:"):
        return default_date

    date_body = date_str[2:]
    match = re.match(r"(\d{14})([+-]\d{2})'(\d{2})'", date_body)
    if match:
        dt_part = match.group(1)
        tz_hour = int(match.group(2))
        tz_minute = int(match.group(3))

        try:
            dt = datetime.strptime(dt_part, "%Y%m%d%H%M%S")
            tz = timezone(timedelta(hours=tz_hour, minutes=tz_minute))
            dt = dt.replace(tzinfo=tz)
            return dt
        except ValueError:
            return default_date

    match_z = re.match(r"(\d{14})Z", date_body)
    if match_z:
        dt_part = match_z.group(1)
        try:
            dt = datetime.strptime(dt_part, "%Y%m%d%H%M%S")
            dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            return default_date

    try:
        if len(date_body) == 8:
            return datetime.strptime(date_body, "%Y%m%d")
        elif len(date_body) == 14:
            return datetime.strptime(date_body, "%Y%m%d%H%M%S")
    except ValueError:
        return default_date

    return default_date


def safe_decode(s: str) -> str:
    """
    decoding word metadata if that needs
    :param s: The raw metadata string to potentially decode.
    :return: The decoded string if the decoding process is successful and the
             result looks more "sensible", otherwise returns the original string.
    """
    if not isinstance(s, str):
        return s
    try:
        return s.encode('latin1').decode('cp1251')
    except UnicodeEncodeError:
        return s


def extract_text_metadata(logger: Logger, file_path: Path, file_format: str) -> tuple:
    """
    Extracts textual content and metadata from a given file.
    :param logger: Logger instance for logging events.
    :param file_path: link of the document
    :param file_format: pdf/doc/djvu
    :return: metainformation about the document
    """
    text = ""
    metadata = dict()
    if file_format == ".pdf":
        doc = fitz.open(file_path)
        text = pdf_to_text(logger, doc)
        metadata["creation_date"] = format_date(doc.metadata.get("creationDate", ""))
        metadata["modification_date"] = format_date(doc.metadata.get("modDate", ""))
        doc.close
    elif file_format in [".docx", ".doc"]:
        if file_format == ".doc":
            file_path = convert_doc_to_docx(file_path)
        with docx2python(file_path) as doc_result:
            all_parts = [
                doc_result.body,
                doc_result.header,
                doc_result.footer
            ]
            raw_metadata = {k: safe_decode(v) for k, v in doc_result.core_properties.items()}
            metadata["creation_date"] = format_date(raw_metadata.get("created", ""))
            metadata["modification_date"] = format_date(raw_metadata.get("modified", ""))
        text = word_to_text(all_parts)
    else:
        logger.warning(f"Unsupported file format: {file_format}")
    return text, metadata


def pdf_to_text(logger: Logger, doc: Document) -> str:
    """
    Extract pdf text: both if it contains text or image
    :param doc: pymupdf Document
    :return: text of document
    """
    full_text = ""
    pages = len(doc)
    logger.info(f"Document contains {pages} pages")
    for page_num in range(pages):
        page = doc.load_page(page_num)
        page_text = page.get_text()
        if len(page_text.strip()) < PDF_SIZE_LIMIT:
            pix = page.get_pixmap(dpi=DPI)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_text = pytesseract.image_to_string(img, lang='rus+eng')
        full_text += page_text + "\n"
    return full_text


def convert_doc_to_docx(file_path: Path) -> str:
    """
    Convert .doc extension to .docx extension
    :param file_path: link of the document
    :return: new path
    """
    new_path = os.path.splitext(file_path)[0] + ".docx"
    doc = aw.Document(file_path)
    doc.save(new_path)
    return new_path


def word_to_text(all_parts: list[list[list[list[list[str]]]]]) -> str:
    """
    Extract text form standard .doc and .docx files
    :param all_parts: list of word doc elements
    :return: text of document
    """
    text_items = []

    def extract_text_recursively(data: str | Iterable) -> None:
        """
        Recursively extracts text from nested structures.
        :param data: A string or an iterable containing strings or further nested iterables.
        """
        if isinstance(data, str):
            if data and data.strip():
                text_items.append(data.strip())
        elif isinstance(data, Iterable):
            for item in data:
                extract_text_recursively(item)

    for part in all_parts:
        extract_text_recursively(part)

    return '\n'.join(filter(None, text_items))
