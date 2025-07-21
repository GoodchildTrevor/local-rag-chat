from typing import Any
from logging import Logger

from functools import lru_cache
from pymorphy2 import MorphAnalyzer


@lru_cache(maxsize=10000)
def get_normal_form(word: str, morph: MorphAnalyzer) -> str:
    """
    Caching of popular lemmas to speed up computation.
    :param word: individual word from query
    :return: lemmatized (normal) form of the word
    """
    return morph.parse(word)[0].normal_form


def extract_entities(results: list) -> tuple[set[str], list[str]]:
    """
    Extract only necessary things from search results.
    :param results: List of result objects
    :return: A tuple of (set of documents, list of file paths)
    """
    docs = set()
    doc_entries = set()

    for hit in results:
        payload = hit.payload  # assuming each result has `.payload` attribute
        doc = payload.get("document", "")
        file_path = payload.get("file_path", "—")

        docs.add(doc)
        doc_entries.add((doc, file_path))

    paths = list({path for _, path in doc_entries})
    return docs, paths


async def search_display(results: tuple[Any, ...], logger: Logger) -> tuple[list | str]:
    """
    Extracting texts for context in prompt and links for message
    :param results: raw search results
    :param logger: Logger instance for logging events.
    :return: texts of docs and their links for chat message
    :raise: ValueError if results list is empty
    """
    if not results:
        logger.warning("No relevant documents")
        raise ValueError("No relevant documents")

    docs, paths = extract_entities(results[:3])
    display_docs = "Релевантные документы:\n" + "\n".join(paths)
    return docs, display_docs


