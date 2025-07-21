import logging

from dataclasses import dataclass
from typing import Optional

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import SparseVector

from config.database import (
    DENSE_VECTOR_CONFIG,
    SPARSE_VECTOR_CONFIG,
    LATE_VECTOR_CONFIG,
)
from config.searching import (
    DENSE_LIMIT,
    SPARSE_LIMIT,
    LATE_LIMIT,
)

logger = logging.getLogger(__name__)


@dataclass
class HybridHit:
    """Search result with structured data."""
    id: str
    score: float
    source: str
    payload: dict


def normalize_scores(scores: list[float]) -> list[float]:
    """
    Normalize scores using min-max normalization.
    :param scores: list of raw scores
    :return: list of normalized scores (0-1 range)
    """
    if not scores:
        raise ValueError("Empty scores list")

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    return [(s - min_score) / (max_score - min_score) for s in scores]


def run_query(
        client: QdrantClient,
        collection: str,
        query: str,
        using: str,
        limit: int
) -> list[models.ScoredPoint]:
    """
    Execute a single query against Qdrant collection.
    :param client: Qdrant client instance
    :param collection: Collection name
    :param query: Query vector (dense or sparse)
    :param using: Vector configuration name
    :param limit: Maximum number of results
    :return: List of scored points from Qdrant
    """
    try:
        response = client.query_points(
            collection_name=collection,
            query=query,
            using=using,
            limit=limit,
            with_payload=True,
        )
        return response.points
    except Exception as e:
        logger.error(f"Query failed for {using}: {e}")
        return list()


def hybrid_search_engine(
        client: QdrantClient,
        collection: str,
        dense_vectors: Optional[list[float]] = None,
        sparse_vectors: Optional[SparseVector] = None,
        late_vectors: Optional[list[float]] = None,
) -> list[models.ScoredPoint]:
    """
    Standard qdrant hybrid search using dense, sparse, and late-interaction vectors.
    :param client: Initialized Qdrant client
    :param collection: Name of the Qdrant collection
    :param dense_vectors: Dense vector for prefetch search
    :param sparse_vectors: Sparse vector for prefetch search
    :param late_vectors: Vector for final reranking (late interaction)
    :return: list of scored points returned by Qdrant
    :raise: ValueError: If no vectors provided or late_vectors missing
    """
    if not any([dense_vectors, sparse_vectors]):
        raise ValueError("At least dense or sparse vectors must be provided")

    if not late_vectors:
        raise ValueError("Late vectors are required for late interaction search")

    prefetch = []

    if dense_vectors:
        prefetch.append(models.Prefetch(
            query=dense_vectors,
            using=DENSE_VECTOR_CONFIG,
            limit=DENSE_LIMIT,
        ))

    if sparse_vectors:
        prefetch.append(models.Prefetch(
            query=sparse_vectors,
            using=SPARSE_VECTOR_CONFIG,
            limit=SPARSE_LIMIT,
        ))

    response = client.query_points(
        collection_name=collection,
        prefetch=prefetch,
        query=late_vectors,
        using=LATE_VECTOR_CONFIG,
        with_payload=True,
        limit=LATE_LIMIT,
    )

    return response.points


def combined_dense_sparse_scores(
        client: QdrantClient,
        collection: str,
        top_k: int,
        threshold: float,
        dense_threshold: float,
        sparse_threshold: float,
        alpha: float = 0.5,
        dense_vectors: Optional[list[float]] = None,
        sparse_vectors: Optional[SparseVector] = None,
) -> list[HybridHit]:
    """
    Custom method: dense and sparse vector search results using weighted score.
    :param client: Initialized Qdrant client
    :param collection: Name of the Qdrant collection
    :param dense_vectors: Dense vector for semantic search
    :param sparse_vectors: Sparse vector for lexical search
    :param top_k: Maximum number of results to return
    :param threshold: Minimum combined score to include a result
    :param dense_threshold: Minimum cosine similarity score to include a result
    :param sparse_threshold: Minimum bm25 score to include a result
    :param alpha: Weight for combining dense and sparse scores (0 to 1)
    :return: list of HybridHit objects with combined and raw scores
    :raise: ValueError: If no vectors provided or alpha out of range
    """
    dense_empty = dense_vectors is None or (hasattr(dense_vectors, '__len__') and len(dense_vectors) == 0)
    sparse_empty = sparse_vectors is None or (hasattr(sparse_vectors, '__len__') and len(sparse_vectors) == 0)

    if dense_empty and sparse_empty:
        raise ValueError("At least one of dense or sparse vectors must be provided")

    dense_hits = [
        hit for hit in run_query(client, collection, dense_vectors, DENSE_VECTOR_CONFIG, DENSE_LIMIT)
        if hit.score >= dense_threshold
    ]

    sparse_hits = [
        hit for hit in run_query(client, collection, sparse_vectors, SPARSE_VECTOR_CONFIG, SPARSE_LIMIT)
        if hit.score >= sparse_threshold
    ]

    if not dense_hits and not sparse_hits:
        logger.warning("No relevant dense or sparse results above thresholds — returning empty list.")
        return list()

    logger.info(f"Dense hits: {len(dense_hits)}, Sparse hits: {len(sparse_hits)}")

    scores: dict[str, dict] = {}

    # Process dense hits (only if not empty)
    if dense_hits:
        dense_raw_scores = {str(hit.id): hit.score for hit in dense_hits}
        dense_scores = normalize_scores([hit.score for hit in dense_hits])

        for i, hit in enumerate(dense_hits):
            sid = str(hit.id)
            scores[sid] = {
                "dense": dense_scores[i],
                "sparse": 0.0,
                "raw_dense": dense_raw_scores[sid],
                "raw_sparse": 0.0,
                "payload": hit.payload,
            }

    # Process sparse hits (only if not empty)
    if sparse_hits:
        sparse_raw_scores = {str(hit.id): hit.score for hit in sparse_hits}
        sparse_scores = normalize_scores([hit.score for hit in sparse_hits])

        for i, hit in enumerate(sparse_hits):
            sid = str(hit.id)
            if sid not in scores:
                scores[sid] = {
                    "dense": 0.0,
                    "raw_dense": 0.0,
                    "payload": hit.payload,
                }
            scores[sid]["sparse"] = sparse_scores[i]
            scores[sid]["raw_sparse"] = sparse_raw_scores[sid]
    # Calculate hybrid scores
    results = []
    for sid, data in scores.items():
        dense_score = data.get("dense", 0.0)
        sparse_score = data.get("sparse", 0.0)
        hybrid_score = alpha * dense_score + (1 - alpha) * sparse_score

        if hybrid_score >= threshold:
            results.append(HybridHit(
                id=sid,
                score=hybrid_score,
                source="dense" if dense_score >= sparse_score else "sparse",
                payload={
                    **data["payload"],
                    "_dense_score": data.get("raw_dense", 0.0),
                    "_sparse_score": data.get("raw_sparse", 0.0),
                }
            ))

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:top_k]
