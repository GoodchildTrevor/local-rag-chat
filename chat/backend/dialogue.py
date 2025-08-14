import asyncio
from qdrant_client import models
from razdel import tokenize
from logging import Logger

from databases.searcher.search import (
    combined_dense_sparse_scores, 
    dense_search, 
    HybridHit
)
from chat.interface.chat_utils import get_normal_form
from config.settings import AppConfig, ClientsConfig, EmbeddingModelsConfig, NLPConfig


class Dialogue:
    """
    Main class for Qdrant DB search
    """
    def __init__(
        self,
        app_config: AppConfig,
        client_config: ClientsConfig,
        embedding_model_config: EmbeddingModelsConfig,
        nlp_config: NLPConfig,
        logger: Logger,
    ):
        self.logger = logger
        self.client = client_config.qdrant_client
        self.dense_embedding_model = embedding_model_config.dense
        self.bm25_embedding_model = embedding_model_config.sparse
        self.dense_threshold = app_config.dense_threshold
        self.sparse_threshold = app_config.sparse_threshold
        self.threshold = app_config.threshold
        self.cosine_similarity_threshold = app_config.cosine_similarity_threshold
        self.top_k = app_config.top_k
        self.morph = nlp_config.morph
        self.stopwords = nlp_config.stopwords

    def processing_query(self, query: str) -> str:
        """
        Normalizing of user's query: tokenize, removing stopwords, lemmatizing
        :param query: user's query
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
    
    async def _vectorize_query(
        self,
        normalized_query: str
    ) -> tuple[list[float], models.SparseVector]:
        """
        One fuction to all vectorizing operations
        :param normalized_query: normalized user's query
        :return: dense and sparse vectors
        """
        dense_future = asyncio.to_thread(
            lambda: next(self.dense_embedding_model.query_embed(normalized_query))
        )
        sparse_future = asyncio.to_thread(
            lambda: next(self.bm25_embedding_model.query_embed(normalized_query))
        )

        dense_vector, sparse_raw = await asyncio.gather(dense_future, sparse_future)
        sparse_vector = models.SparseVector(indices=sparse_raw.indices, values=sparse_raw.values)

        self.logger.debug("Embeddings computed")
        return dense_vector, sparse_vector

    async def get_searching_results(
            self, 
            collection:str, 
            normalized_query: str
    ) -> list[HybridHit]:
        """
        Vectorizes the user's query and retrieves relevant search results.
        1. Normalizes and processes the input query.
        2. Generates a dense vector using a dense embedding model.
        3. Generates a sparse vector using a BM25-based embedding model.
        4. Combines both dense and sparse representations to retrieve the most relevant text chunks.
        :param normalized_query: normalized user's query
        :return: a list of top-matching text chunks based on the combined embedding scores.
        """
        dense_vector, sparse_vector = await self._vectorize_query(normalized_query)
        results = combined_dense_sparse_scores(
            client=self.client,
            collection=collection,
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
    
    async def get_cashed_answers(
            self, 
            collection:str, 
            normalized_query: str
    ) -> list[HybridHit]:
        """
        Vectorizes the user's query and retrieves relevant search results.
        1. Normalizes and processes the input query.
        2. Check cashed questions for similar examples
        :param collection:
        :param normalized_query: normalized user's query
        :return: a list consists of one most similar cashed question (if similarity is high enough)
        """
        dense_vector, _ = await self._vectorize_query(normalized_query)
        return dense_search(
            client=self.client,
            collection=collection,
            dense_vectors=dense_vector,
            cosine_similarity_threshold=self.cosine_similarity_threshold
        )
