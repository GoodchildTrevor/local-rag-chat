import aiohttp
from logging import Logger
from pathlib import Path
import re
from typing import Any, Pattern
from docx2python import docx2python
from config.settings import AppConfig


async def word_to_text(
    logger: Logger,
    app_config: AppConfig,
    file_path: Path,
    session: aiohttp.ClientSession
) -> list[dict[str, Any]]:
    """
    Extract text and tables from a Word document.
    Processes both plain text and tables, handling image placeholders.
    All content types return consistent structure with 'type' and 'content' keys.

    :param logger: Logger instance for logging messages
    :param app_config: Application configuration object.
    :param file_path: Path to the Word document
    :param session: aiohttp ClientSession for making requests.
    :return: List of document elements, each with 'type' (str) and 'content' (list[str])
    """
    elements: list[dict[str, Any]] = []
    doc_result = docx2python(file_path, html=True)
    try:
        img_pattern: Pattern[str] = re.compile(r"----(?:media/)?(image\d+\.\w+)----")
        for section_idx, section in enumerate(doc_result.body):
            for item_idx, item in enumerate(section):
                if is_table_structure(item):
                    table_marker = f"[TABLE_{section_idx}]"
                    table_rows = extract_table_data(item)
                    if item_idx == 0:
                        headers = table_rows
                        formatted_rows = format_table(headers, table_rows)
                        if formatted_rows:
                            elements.append({
                                "type": "table",
                                "content": formatted_rows,
                                "_meta": {
                                    "table_marker": table_marker,
                                    "item_idx": item_idx
                                }
                            })
                        continue
                if isinstance(item, list):
                    flat_text: str = " ".join(flatten(item)).strip()
                    if flat_text:
                        clean_text = clean_html(flat_text)
                        if clean_text:
                            images_in_text = img_pattern.findall(clean_text)
                            if images_in_text:
                                parts = re.split(img_pattern, clean_text)
                                if parts and parts[0].strip():
                                    elements.append({
                                        "type": "text",
                                        "content": [parts[0].strip()]
                                    })
                                for img_idx, img_name in enumerate(images_in_text):
                                    img_data = doc_result.images.get(img_name)
                                    img_text = ""
                                    if img_data:
                                        try:
                                            form = aiohttp.FormData()
                                            form.add_field(
                                                'file',
                                                img_data,
                                                filename=img_name,
                                            )
                                            async with session.post(
                                                app_config.file_worker,
                                                data=form,
                                                timeout=aiohttp.ClientTimeout(total=600)
                                            ) as response:
                                                if response.status == 200:
                                                    img_text = await response.text()
                                                    # Проверяем, что текст не пустой
                                                    if not img_text or not img_text.strip():
                                                        img_text = f"[Image: {img_name} - empty response]"
                                                else:
                                                    logger.error(f"Image service returned status {response.status} for {img_name}")
                                        except aiohttp.ClientError as e:
                                            logger.error(f"Failed to process image {img_name}: {e}")
                                            img_text = f"[Image: {img_name} - network error]"
                                        except Exception as e:
                                            logger.error(f"Unexpected error processing image {img_name}: {e}")
                                            img_text = f"[Image: {img_name} - processing error]"
                                    elements.append({
                                        "type": "image",
                                        "content": [img_text]
                                    })
                                    if img_idx + 1 < len(parts) and parts[img_idx + 1].strip():
                                        elements.append({
                                            "type": "text",
                                            "content": [parts[img_idx + 1].strip()]
                                        })
                            else:
                                elements.append({
                                    "type": "text",
                                    "content": [clean_text]
                                })
    finally:
        doc_result.close()
    logger.info(f"Extracted {len(elements)} elements from document")
    text_count = sum(1 for el in elements if el["type"] == "text")
    table_count = sum(1 for el in elements if el["type"] == "table")
    image_count = sum(1 for el in elements if el["type"] == "image")
    logger.info(f"Elements breakdown: {text_count} text, {table_count} tables, {image_count} images")
    return elements


def merge_table_parts(parts: list[list[list[str]]]) -> list[list[str]]:
    """
    Merge multiple table parts into a single table by concatenating columns.
    :param parts: List of table parts to merge.
    :return: Merged table as a list of rows.
    """
    if not parts:
        return []
    # Single part requires no merging
    if len(parts) == 1:
        return parts[0]
    # Determine maximum number of rows across all parts
    max_rows = max(len(part) for part in parts)
    # Merge columns horizontally
    merged: list[list[str]] = []
    for row_idx in range(max_rows):
        merged_row: list[str] = []
        for part in parts:
            if row_idx < len(part):
                merged_row.extend(part[row_idx])
            else:
                # Pad with empty cells for missing rows
                if part:
                    merged_row.extend([""] * len(part[0]))
        merged.append(merged_row)
    return merged


def is_table_structure(item: Any) -> bool:
    """
    Determine if a nested list structure represents a table based on its depth and consistency.
    :param item: Input item to check.
    :return: True if the structure resembles a table, False otherwise.
    """
    if not isinstance(item, list):
        return False
    # Check for sufficient nesting depth (tables typically have multiple nested levels)
    if len(item) > 0 and isinstance(item[0], list):
        # Require at least 2 rows for a table
        if len(item) >= 2:
            row_lengths: list[int] = []
            for row in item:
                if isinstance(row, list):
                    # Count cells in each row
                    cells = [cell for cell in row if isinstance(cell, (str, list))]
                    row_lengths.append(len(cells))
            # If multiple rows with similar cell counts, likely a table
            if len(row_lengths) >= 2:  # and len(set(row_lengths)) <= 2:
                return True
    return False


def extract_table_data(item: list[Any]) -> list[list[str]]:
    """
    Extract table data from a nested list structure into a list of rows.
    :param item: Nested list structure representing a table.
    :return: List of rows, each row is a list of cell strings.
    """
    rows: list[list[str]] = []
    for row in item:
        if isinstance(row, list):
            cells: list[str] = []
            for cell in row:
                if isinstance(cell, list):
                    cell_text: str = " ".join(flatten(cell)).strip()
                else:
                    cell_text = str(cell).strip()
                cell_text = clean_html(cell_text)
                cell_text = cell_text.replace("\n", " ").replace("\r", " ").strip()
                cell_text = re.sub(r'\s+', ' ', cell_text)
                if cell_text:
                    cells.append(cell_text)
            if cells:
                rows.append(cells)
    return rows


def flatten(item: Any) -> list[str]:
    """
    Recursively flatten nested lists into a flat list of strings.
    :param item: Input item to flatten. Can be a string or nested list structure.
    :return: Flat list of strings extracted from the input.
    """
    if isinstance(item, str):
        return [item]
    result: list[str] = []
    for sub in item:
        result.extend(flatten(sub))
    return result


def clean_html(text: str) -> str:
    """
    Remove HTML tags and watermarks from text.
    Specifically removes Aspose.Words evaluation copy notices and HTML tags.
    :param text: Input text containing HTML.
    :return: Cleaned text without HTML tags or watermarks.
    """
    text = re.sub(
        r"Created with an evaluation copy of Aspose\.Words\..*?</span>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r'<a href="https://products\.aspose\.com/words/temporary-license/">.*?</a>',
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def format_table(headers: list[list[str]], table_data: list[list[str]]) -> list[str]:
    """
    Format table rows into human-readable key-value strings.
    Handles:
    - Standard multi-column tables: uses first row as headers
    - Single-column glossaries: treats first row as column name, rest as values

    :param headers: List of header rows.
    :param table_data: List of rows; each row is a list of cell strings.
    :return: List of formatted strings like "Header1: Value1 | Header2: Value2"
    """
    if not table_data:
        return []
    if not headers:
        return []
    if len(headers) == 0 or len(headers[0]) == 0:
        return []
    if len(table_data) == 0 or len(table_data[0]) == 0:
        return []

    # Handle single-column tables (glossaries)
    if len(headers[0]) == 1 and len(table_data[0]) == 1:
        formatted = []
        for row in table_data:
            if row and row[0]:
                formatted.append(row[0].strip())
        return formatted

    # Handle multi-column tables
    formatted = []
    for row in table_data:
        if not row:
            continue
        parts = []
        for col_idx, cell in enumerate(row):
            if col_idx < len(headers[0]) and headers[0][col_idx]:
                header_text = headers[0][col_idx].strip()
                cell_text = cell.strip() if cell else ""
                if cell_text:
                    parts.append(f"{header_text}: {cell_text}")
            elif cell and cell.strip():
                parts.append(cell.strip())
        if parts:
            formatted.append(" | ".join(parts))
    return formatted


async def convert_doc_to_docx(
    doc_path: Path,
    logger: Logger,
    timeout: int = 60,
) -> str | None:
    """
    Convert .doc to .docx using LibreOffice.

    Uses a dedicated temporary directory as the LibreOffice output target so
    the expected output file can be located by exact name, avoiding accidental
    glob matches on similarly-named files in the source directory.
    """
    doc_path = Path(doc_path)

    if not shutil.which("libreoffice"):
        logger.error("LibreOffice not found. Required for .doc conversion.")
        return None

    tmp_dir = tempfile.mkdtemp()
    try:
        proc = await asyncio.create_subprocess_exec(
            "libreoffice",
            "--headless",
            "--convert-to",
            "docx",
            "--outdir",
            tmp_dir,
            str(doc_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            logger.error(
                "LibreOffice conversion timed out after %d seconds", timeout
            )
            return None

        if proc.returncode != 0:
            stderr_text = stderr.decode()[:200] if stderr else "unknown error"
            logger.error("LibreOffice conversion failed: %s", stderr_text)
            return None

        expected = Path(tmp_dir) / f"{doc_path.stem}.docx"
        if not expected.exists():
            logger.error(
                "LibreOffice succeeded but output not found: %s", expected
            )
            return None

        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as out:
            out_path = out.name
        shutil.move(str(expected), out_path)
        return out_path

    except Exception as e:
        logger.error("Unexpected conversion error: %s", e, exc_info=True)
        return None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
