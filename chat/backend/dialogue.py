"""Dialogue: orchestrates NLP query normalisation + remote vector search."""
from __future__ import annotations

from logging import Logger

from razdel import tokenize

from chat.interface.chat_utils import get_normal_form
from config.settings import AppConfig, ClientsConfig, NLPConfig
from databases.searcher.searcher_client import HybridHit, SearcherClient
from databases.cashing.cashing import AnswerCash


class Dialogue:
    """
    Main class for user-facing search.

    Query lifecycle:
      1. processing_query()  — tokenise + lemmatise (local NLP, pymorphy3)
      2. get_searching_results() — POST qdrant-searcher /vector_search
      3. get_cached_answers()    — POST qdrant-searcher /vector_search on cache collection
    """

    def __init__(
        self,
        app_config: AppConfig,
        clients_config: ClientsConfig,
        nlp_config: NLPConfig,
        logger: Logger,
    ) -> None:
        self.app_config = app_config
        self.logger = logger
        # Redis client kept for AnswerCash
        self.redis = clients_config.redis_client
        self.morph = nlp_config.morph
        self.stopwords = nlp_config.stopwords
        self.top_k = app_config.top_k
        self.cosine_similarity_threshold = app_config.cosine_similarity_threshold

        self._searcher = SearcherClient(
            base_url=app_config.qdrant_searcher_url,
            timeout=30.0,
        )

    # ------------------------------------------------------------------
    # NLP
    # ------------------------------------------------------------------

    def processing_query(self, query: str) -> str:
        """
        Tokenise, remove stopwords, lemmatise — all local (pymorphy3 + razdel).
        """
        tokens = [
            t.text.lower() for t in tokenize(query)
            if t.text.isalpha()
            and t.text.lower() not in self.stopwords
            and len(t.text) > 1
        ]
        lemmas = [get_normal_form(tok, self.morph) for tok in tokens]
        normalized = " ".join(lemmas).strip()
        self.logger.info("Query: `%s` → normalized `%s`", query, normalized)
        return normalized

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def get_searching_results(
        self,
        collection: str,
        normalized_query: str,
    ) -> list[HybridHit]:
        """
        Hybrid vector search via qdrant-searcher microservice.
        """
        return await self._searcher.search(
            text=normalized_query,
            collection_name=collection,
            method="hybrid",
            top_k=self.top_k,
        )

    async def get_cached_answers(
        self,
        collection: str,
        normalized_query: str,
    ) -> list[HybridHit]:
        """
        Dense-only search against the cache collection.
        Returns at most one hit (best match above threshold).
        """
        hits = await self._searcher.search(
            text=normalized_query,
            collection_name=collection,
            method="dense",
            top_k=1,
        )
        filtered = [
            h for h in hits
            if h.score >= self.cosine_similarity_threshold
        ]
        if not filtered:
            self.logger.warning("No cached answer above threshold for query: %s", normalized_query)
        return filtered
