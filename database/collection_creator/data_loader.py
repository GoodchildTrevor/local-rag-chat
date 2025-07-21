import time
import uuid
import logging
from more_itertools import chunked
from pathlib import Path
from typing import Any, List, Dict

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def upsert_data(
    client: QdrantClient,
    collection_name: str,
    dense_embeddings: List[List[float]],
    bm25_embeddings: List[Any],  # SparseVectorObject
    late_interaction_embeddings: List[List[float]],
    name: str,
    documents: List[str],
    metadata: Dict[str, Any],
    file_path: str,
    batch_size: int = 16,
) -> None:
    """
    Upsert points into Qdrant in batches with logging.
    :param client: QdrantClient instance
    :param collection_name: name of the Qdrant collection
    :param dense_embeddings: list of dense vectors
    :param bm25_embeddings: list of sparse vector objects (with .as_object())
    :param late_interaction_embeddings: list of late-interaction vectors
    :param name: name of doc
    :param documents: original chunk texts
    :param metadata: metadata dictionary for all chunks
    :param file_path: source file path (for payload)
    :param batch_size: number of points per upsert batch
    """
    total_points = len(dense_embeddings)
    logger.info(f"Preparing to upsert {total_points} points from {file_path}")

    # Build PointStruct list
    points = []
    for dense, sparse_obj, late, doc in zip(
        dense_embeddings,
        bm25_embeddings,
        late_interaction_embeddings,
        documents,
    ):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                settings.dense_vector_config: dense,
                settings.sparse_vector_config: sparse_obj.as_object(),
                settings.late_vector_config: late,
            },
            payload={
                "name": name,
                "document": doc,
                "metadata": metadata,
                "file_path": file_path,
            },
        )
        points.append(point)
    # Upsert in batches
    for batch in chunked(points, batch_size):
        try:
            client.upsert(
                collection_name=collection_name,
                points=batch
            )
            logger.debug(f"Upserted batch of {len(batch)} points into '{collection_name}'")

            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Failed to upsert batch: {e}")

    logger.info(f"Completed upsert of {total_points} points into '{collection_name}'")


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
