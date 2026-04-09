import asyncio
import uuid
import logging
from pathlib import Path
from typing import Any, Optional

from more_itertools import chunked
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse

from config.settings import EmbeddingModelsConfig

logger = logging.getLogger(__name__)


async def upsert_data(
    client: AsyncQdrantClient,
    collection_name: str,
    embedding_model_config: EmbeddingModelsConfig,
    dense_embeddings: Optional[list[list[float]]],
    bm25_embeddings: Optional[list[Any]],  # SparseVectorObject with .as_object()
    late_interaction_embeddings: Optional[list[list[float]]],
    base_payload: dict[str, Any],
    chunks: list[dict[str, Any]],
    batch_size: int = 16,
) -> None:
    """
    Upsert points into Qdrant in batches with logging.

    :param client: AsyncQdrantClient instance.
    :param collection_name: Name of the Qdrant collection.
    :param embedding_model_config: Configuration object for vector names.
    :param dense_embeddings: list of dense vectors.
    :param bm25_embeddings: list of sparse vector objects (with .as_object()).
    :param late_interaction_embeddings: list of late-interaction vectors.
    :param base_payload: File-level metadata shared across all chunks
           (e.g. file_path, file_name, creation_date, modification_date).
    :param chunks: list of chunk dicts with 'raw', 'lemmas', and '_meta' keys.
    :param batch_size: Number of points per upsert batch.
    """
    embeddings_map: dict[str, list[Any]] = {}

    if dense_embeddings is not None:
        embeddings_map[embedding_model_config.dense_vector_config] = dense_embeddings
    if bm25_embeddings is not None:
        embeddings_map[embedding_model_config.sparse_vector_config] = bm25_embeddings
    if late_interaction_embeddings is not None:
        embeddings_map[embedding_model_config.late_vector_config] = late_interaction_embeddings

    if not embeddings_map:
        logger.warning("No embeddings provided. Skipping upsert.")
        return

    def point_generator():
        for i, chunk in enumerate(chunks):
            vector_dict: dict[str, Any] = {}

            for name, embeds in embeddings_map.items():
                if name == embedding_model_config.sparse_vector_config:
                    vector_dict[name] = embeds[i].as_object()
                else:
                    vector_dict[name] = embeds[i]

            meta = chunk.get("_meta") or {}

            point_payload = {
                **base_payload,
                "document": chunk["raw"],
                "page_start": meta.get("page_start"),
                "page_end": meta.get("page_end"),
                "table_marker": meta.get("table_marker"),
                "row_index": meta.get("row_index"),
                "chunk_tokens": meta.get("tokens"),
            }
            # Remove None values
            point_payload = {k: v for k, v in point_payload.items() if v is not None}

            yield PointStruct(
                id=str(uuid.uuid4()),
                vector=vector_dict,
                payload=point_payload,
            )

    upserted_points = 0
    for batch in chunked(point_generator(), batch_size):
        try:
            await client.upsert(
                collection_name=collection_name,
                points=list(batch),
                wait=True,
            )
            upserted_points += len(batch)
            logger.debug(
                "Upserted batch of %d points into '%s'",
                len(batch),
                collection_name,
            )
            await asyncio.sleep(0.5)
        except Exception as e:
            logger.error("Failed to upsert batch: %s", e)

    logger.info("Completed upsert of %d points into '%s'", upserted_points, collection_name)


async def sync_file_paths(
    client: AsyncQdrantClient,
    collection: str,
    current_file_paths: set[Path],
    payload_key: str = "file_path",
    scroll_limit: int = 1000,
) -> tuple[set[Path], set[str]]:
    """
    Perform a single pass over the Qdrant collection to synchronize file paths
    between the vector store and the current filesystem state.

    :param client: An initialized AsyncQdrantClient instance.
    :param collection: Name of the Qdrant collection to scan.
    :param current_file_paths: set of Path objects representing existing files on disk.
    :param payload_key: Payload field name in Qdrant storing the file path. Defaults to "file_path".
    :param scroll_limit: Maximum number of points to retrieve per scroll request. Defaults to 1000.
    :return: A tuple (new_paths, deleted_paths).
    :raises UnexpectedResponse: If an unexpected error occurs during scrolling.
    """
    current_paths_str = {str(p) for p in current_file_paths}
    db_paths_str: set[str] = set()

    offset = None
    try:
        while True:
            points, offset = await client.scroll(
                collection_name=collection,
                with_payload=[payload_key],
                with_vectors=False,
                limit=scroll_limit,
                offset=offset,
            )
            if not points:
                break

            for point in points:
                path = point.payload.get(payload_key)
                if path:
                    db_paths_str.add(path)

            if not offset:
                break

    except UnexpectedResponse as e:
        if getattr(e, "response", None) and e.response.status_code == 404:
            logger.warning("Collection '%s' not found. Assuming it is empty.", collection)
            return current_file_paths, set()
        logger.error("Unexpected error during collection scan: %s", e)
        raise

    new_paths = {Path(p) for p in (current_paths_str - db_paths_str)}
    deleted_paths = db_paths_str - current_paths_str

    logger.info(
        "Sync result: %d new files to ingest, %d deleted files to clean up.",
        len(new_paths),
        len(deleted_paths),
    )
    return new_paths, deleted_paths


async def delete_orphaned_chunks(
    client: AsyncQdrantClient,
    collection_name: str,
    deleted_file_paths: set[str],
    payload_key: str = "file_path",
    scroll_limit: int = 1000,
) -> int:
    """
    Delete all vector points associated with file paths that no longer exist on disk.

    :param client: An initialized AsyncQdrantClient instance.
    :param collection_name: Name of the Qdrant collection to clean.
    :param deleted_file_paths: set of file path strings missing from the filesystem.
    :param payload_key: Payload field name storing the file path. Defaults to "file_path".
    :param scroll_limit: Number of points to fetch per scroll request. Defaults to 1000.
    :return: Number of deleted points.
    """
    if not deleted_file_paths:
        return 0

    point_ids = []
    offset = None

    while True:
        points, offset = await client.scroll(
            collection_name=collection_name,
            with_payload=[payload_key],
            with_vectors=False,
            limit=scroll_limit,
            offset=offset,
        )
        if not points:
            break

        for pt in points:
            if pt.payload.get(payload_key) in deleted_file_paths:
                point_ids.append(pt.id)

        if not offset:
            break

    if point_ids:
        for batch in chunked(point_ids, 1000):
            await client.delete(
                collection_name=collection_name,
                points_selector=batch,
                wait=True,
            )
        logger.info(
            "🧹 Deleted %d orphaned chunks from %d missing files.",
            len(point_ids),
            len(deleted_file_paths),
        )
        return len(point_ids)

    return 0
