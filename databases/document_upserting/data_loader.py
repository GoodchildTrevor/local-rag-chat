import time
import uuid
from uuid import UUID
import logging
from more_itertools import chunked
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse

from config.settings import EmbeddingModelsConfig

logger = logging.getLogger(__name__)


def upsert_data(
    client: QdrantClient,
    collection_name: str,
    embeding_model_config: EmbeddingModelsConfig,
    dense_embeddings: list[list[float]],
    bm25_embeddings: list[Any] | None,  # SparseVectorObject
    late_interaction_embeddings: list[list[float]] | None,
    payload: dict,
    documents: list[str],
    batch_size: int = 16,
) -> None:
    """
    Upsert points into Qdrant in batches with logging.
    :param client: QdrantClient instance
    :param collection_name: name of the Qdrant collection
    :param dense_embeddings: list of dense vectors
    :param bm25_embeddings: list of sparse vector objects (with .as_object())
    :param late_interaction_embeddings: list of late-interaction vectors
    :param payload: additional info about text
    :param documents: original chunk texts
    :param batch_size: number of points per upsert batch
    """
    embeddings_map = {}
    if dense_embeddings is not None:
        embeddings_map[embeding_model_config.dense_vector_config] = dense_embeddings
    if bm25_embeddings is not None:
        embeddings_map[embeding_model_config.sparse_vector_config] = bm25_embeddings
    if late_interaction_embeddings is not None:
        embeddings_map[embeding_model_config.late_vector_config] = late_interaction_embeddings

    if not embeddings_map:
        logger.warning("No embeddings provided. Skipping upsert.")
        return

    def point_generator():
        for i, doc in enumerate(documents):
            vector_dict = {}
            for name, embeds in embeddings_map.items():
                if name == embeding_model_config.sparse_vector_config:
                    vector_dict[name] = embeds[i].as_object()
                else:
                    vector_dict[name] = embeds[i]

            point_payload = payload.copy()
            point_payload["document"] = doc
            id = point_payload.get("question_id", str(uuid.uuid4()))

            yield PointStruct(
                id=id,
                vector=vector_dict,
                payload=point_payload,
            )

    upserted_points = 0
    for batch in chunked(point_generator(), batch_size):
        try:
            client.upsert(
                collection_name=collection_name,
                points=batch,
                wait=True
            )
            upserted_points += len(batch)
            logger.debug(f"Upserted batch of {len(batch)} points into '{collection_name}'")

            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Failed to upsert batch: {e}")

    logger.info(f"Completed upsert of {upserted_points} points into '{collection_name}'")


def get_new_file_paths(
    client: QdrantClient,
    collection: str,
    target_paths: set[Path],
    scroll_limit: int,
    payload_key: str = "file_path",
    offset=None,
) -> set[Path]:
    """
    The function scrolls through the collection and checks which of the provided paths already exist
    based on their payload field. It returns only the paths that are not found in the collection.
    :param client: Qdrant client instance used to connect to the vector store.
    :param collection: Name of the Qdrant collection to search within.
    :param target_paths: Set of file paths to verify.
    :param scroll_limit: limit of chunks in one scroll
    :param payload_key: Payload key in Qdrant that holds the file path value. Defaults to "file_path".
    :param offset: offset in scroll
    :return: Set of file paths that are new (not found in the collection).
    :raises Exception: If an unexpected error occurs while querying the collection.
    """
    known_paths = set()
    remaining_paths = {str(p) for p in target_paths}

    try:
        while remaining_paths:
            response = client.scroll(
                collection_name=collection,
                with_payload=[payload_key],
                limit=scroll_limit,
                offset=offset
            )
            points, offset = response
            if not points:
                break

            for point in points:
                path = point.payload.get(payload_key)
                if path in remaining_paths:
                    known_paths.add(Path(path))
                    remaining_paths.remove(path)

            if not offset:
                break

    except UnexpectedResponse as e:
        if hasattr(e, 'response') and e.response.status_code == 404:
            logger.warning(f"Collection '{collection}' not found. Assuming it's empty.")
            return target_paths
        else:
            logger.error(f"Unexpected error during scroll: {e}")
            raise Exception("Unexpected error while scrolling Qdrant collection.") from e

    return target_paths - known_paths
