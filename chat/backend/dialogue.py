from typing import Any

import asyncio
from qdrant_client import models
from razdel import tokenize

from database.searcher.search import combined_dense_sparse_scores
from chat.interface.chat_utils import get_normal_form
from config.settings import get_settings

settings = get_settings()


class Search:

    def __init__(
        self,
        logger,
        collection,
    ):
        self.logger = logger
        self.client = settings.client
        self.collection = collection
        self.dense_embedding_model = settings.dense_embedding_model
        self.bm25_embedding_model = settings.bm25_embedding_model
        self.dense_threshold = settings.dense_threshold
        self.sparse_threshold = settings.sparse_threshold
        self.threshold = settings.threshold
        self.top_k = settings.top_k
        self.morph = settings.morph
        self.stopwords = settings.stopwords

    def processing_query(self, query) -> str:
        """
        Normalizing of user's query: tokenize, removing stopwords, lemmatizing
        :param: user's query
        :return: normalized query
        """
        tokens = [
            t.text.lower() for t in tokenize(query)
            if t.text.isalpha() and t.text.lower() not in self.stopwords and len(t.text) > 1
        ]
        lemmas = [get_normal_form(tok, self.morph) for tok in tokens]
        normalized_query = " ".join(lemmas).strip()
        self.logger.info(f"Query: `{query}` → normalized `{normalized_query}`")
        return normalized_query

    async def get_searching_results(self, query: str) -> tuple[Any, ...]:
        """
        Vectorizes the user's query and retrieves relevant search results.
        1. Normalizes and processes the input query.
        2. Generates a dense vector using a dense embedding model.
        3. Generates a sparse vector using a BM25-based embedding model.
        4. Combines both dense and sparse representations to retrieve the most relevant text chunks.
        :param: user's query
        :return: a list of top-matching text chunks based on the combined embedding scores.
        """
        normalized_query = self.processing_query(query)
        dense_vector = await asyncio.to_thread(
            lambda q=normalized_query: next(self.dense_embedding_model.query_embed(q))
        )
        sparse_raw = await asyncio.to_thread(
            lambda q=normalized_query: next(self.bm25_embedding_model.query_embed(q))
        )
        sparse_vector = models.SparseVector(indices=sparse_raw.indices, values=sparse_raw.values)
        self.logger.debug("Embeddings computed")
        results = combined_dense_sparse_scores(
            client=self.client,
            collection=self.collection,
            dense_vectors=dense_vector,
            sparse_vectors=sparse_vector,
            dense_threshold=self.dense_threshold,
            sparse_threshold=self.sparse_threshold,
            threshold=self.threshold,
            top_k=self.top_k
        )

        for hit in results:
            score = hit.score
            doc_id = hit.id
            source = hit.source
            payload = hit.payload or {}
            dense_score = payload.get("_dense_score")
            sparse_score = payload.get("_sparse_score")
            snippet = payload.get("document", "")[:100].replace("\n", " ")
            path = payload.get("file_path", "—")

            self.logger.info(
                f"[{source.upper():6}] score={score:7.4f}  "
                f"dense={dense_score if dense_score is not None else '  N/A':>7}  "
                f"sparse={sparse_score if sparse_score is not None else '  N/A':>7}  "
                f"id={doc_id}  path={path}  text='{snippet}…'"
            )

        return results
