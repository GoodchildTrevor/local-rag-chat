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

import pymorphy2
from razdel import sentenize, tokenize
import tiktoken
from stop_words import get_stop_words

from config.database import (
    PDF_SIZE_LIMIT,
    DPI
)

RU_STOPWORDS = set(get_stop_words("ru"))
morph = pymorphy2.MorphAnalyzer()
tokenizer = tiktoken.get_encoding("cl100k_base")


def chunker(text: str, max_tokens: int, overlap: int) -> list[dict]:
    """
    Custom chunker for documents
    :param text: raw text
    :param max_tokens: max tokens in chunk
    :param overlap: default 1 sentence
    :return: list of chunks
    """
    sentences = list(sentenize(text))

    processed = []
    for s in sentences:
        raw = s.text
        tokens = [
            t.text.lower() for t in tokenize(raw)
            if t.text.isalpha() and t.text.lower() not in RU_STOPWORDS and len(t.text) > 1
        ]
        lemmas = [morph.parse(tok)[0].normal_form for tok in tokens]
        lemmatized = " ".join(lemmas)
        processed.append({"raw": raw, "lemmas": lemmatized})

    chunks = []
    current_chunk_raw = []
    current_chunk_lemmas = []
    current_token_count = 0
    i = 0

    while i < len(processed):
        sent = processed[i]
        lemma_sent = sent["lemmas"]
        token_count = len(tokenizer.encode(lemma_sent, disallowed_special=()))

        if current_token_count + token_count > max_tokens:
            if current_chunk_raw:
                chunks.append({
                    "raw": " ".join(current_chunk_raw),
                    "lemmas": " ".join(current_chunk_lemmas),
                })
            i = max(0, i - overlap)
            current_chunk_raw = []
            current_chunk_lemmas = []
            current_token_count = 0
        else:
            current_chunk_raw.append(sent["raw"])
            current_chunk_lemmas.append(lemma_sent)
            current_token_count += token_count
            i += 1

    if current_chunk_raw:
        chunks.append({
            "raw": " ".join(current_chunk_raw),
            "lemmas": " ".join(current_chunk_lemmas),
        })

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
        text = pdf_to_text(doc)
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


def pdf_to_text(doc: Document) -> str:
    """
    Extract pdf text: both if it contains text or image
    :param doc: pymupdf Document
    :return: text of document
    """
    full_text = ""
    for page_num in range(len(doc)):
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
