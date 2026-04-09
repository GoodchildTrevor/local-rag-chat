import asyncio
import gc
import logging
import os
import random
import statistics
import tracemalloc
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import aiohttp
import psutil
from more_itertools import chunked

from config.settings import (
    AppConfig,
    ClientsConfig,
    DBConfig,
    EmbeddingModelsConfig,
    NLPConfig,
)
from databases.document_upserting.data_loader import (
    delete_orphaned_chunks,
    sync_file_paths,
    upsert_data,
)
from databases.document_upserting.data_processing import (
    chunker,
    extract_text_metadata,
)

LOG_DIR = "/app/logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "qdrant_ingest.log")

log_format = "[%(asctime)s] %(levelname)s - %(message)s"
formatter = logging.Formatter(log_format)

file_handler = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
if not root_logger.handlers:
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)


async def wait_for_memory(
    logger_instance: logging.Logger,
    max_ram_fraction: float = 0.5,
    poll_interval: float = 2.0,
) -> None:
    """
    Block until current RAM usage drops below ``max_ram_fraction``.

    :param logger_instance: Logger instance.
    :param max_ram_fraction: Maximum allowed fraction of RAM usage (0.5 = 50%).
    :param poll_interval: Seconds to wait between checks.
    """
    while True:
        current_pct = psutil.virtual_memory().percent / 100.0
        if current_pct < max_ram_fraction:
            return
        logger_instance.warning(
            "RAM at %.1f%% (limit %.0f%%), waiting %.1fs before next task...",
            current_pct * 100,
            max_ram_fraction * 100,
            poll_interval,
        )
        await asyncio.sleep(poll_interval)


def compute_semaphore_limit(
    target_fraction: float = 0.5,
    estimated_mb_per_doc: float = 500.0,
    min_concurrency: int = 1,
) -> int:
    """
    Compute a safe concurrency limit based on available RAM.

    :param target_fraction: Fraction of total RAM to allocate for all concurrent tasks.
    :param estimated_mb_per_doc: Estimated memory (in MB) required per task.
    :param min_concurrency: Minimum number of concurrent tasks to allow.
    :return: Maximum number of tasks to run simultaneously.
    """
    total_mb = psutil.virtual_memory().total / (1024 ** 2)
    budget_mb = total_mb * target_fraction
    return max(min_concurrency, int(budget_mb / estimated_mb_per_doc))


async def process_collection(
    session: aiohttp.ClientSession,
    collection_name: str,
    file_paths: list[Path],
    app_config: AppConfig,
    client_config: ClientsConfig,
    db_config: DBConfig,
    embedding_model_config: EmbeddingModelsConfig,
    nlp_config: NLPConfig,
) -> None:
    """
    Process a single collection: synchronize file paths, delete orphaned chunks,
    and upsert new documents.

    :param session: Shared aiohttp client session.
    :param collection_name: Name of the collection in Qdrant.
    :param file_paths: List of file paths belonging to the collection.
    :param app_config: Application configuration.
    :param client_config: Clients configuration.
    :param db_config: Database configuration.
    :param embedding_model_config: Embedding models configuration.
    :param nlp_config: NLP configuration.
    """
    logger.info("Processing collection: %s", collection_name)

    current_file_paths = set(file_paths)

    paths_for_etl, deleted_paths = await sync_file_paths(
        client=client_config.qdrant_client,
        collection=collection_name,
        current_file_paths=current_file_paths,
        payload_key="file_path",
        scroll_limit=db_config.scroll_limit,
    )

    logger.info("[%s] %d documents for deletion", collection_name, len(deleted_paths))
    if deleted_paths:
        await delete_orphaned_chunks(
            client=client_config.qdrant_client,
            collection_name=collection_name,
            deleted_file_paths=deleted_paths,
            payload_key="file_path",
            scroll_limit=db_config.scroll_limit,
        )

    logger.info("[%s] %d documents for upsert", collection_name, len(paths_for_etl))

    dense_model = embedding_model_config.dense
    sparse_model = embedding_model_config.sparse

    for file_path in paths_for_etl:
        file_format = file_path.suffix.lower()
        if file_format not in db_config.file_format:
            logger.debug(
                "[%s] Skipping unsupported format: %s",
                collection_name,
                file_path,
            )
            continue

        logger.info("[%s] Processing file: %s", collection_name, file_path.name)

        try:
            elements, metadata = await extract_text_metadata(
                logger,
                app_config,
                file_path,
                file_format,
                nlp_config,
                session,
            )

            chunks = await asyncio.to_thread(
                chunker,
                logger,
                nlp_config,
                elements,
                db_config.chunk_size,
                db_config.overlap,
            )
            lemma_texts = [c["lemmas"] for c in chunks]
            logger.info(
                "[%s] %s: %d chunks generated",
                collection_name,
                file_path.name,
                len(chunks),
            )

            dense_embeddings, sparse_embeddings = [], []
            for batch in chunked(lemma_texts, db_config.batch_size):
                dense_batch, sparse_batch = await asyncio.gather(
                    asyncio.to_thread(dense_model.embed, batch),
                    asyncio.to_thread(sparse_model.embed, batch),
                )
                dense_embeddings.extend(dense_batch)
                sparse_embeddings.extend(sparse_batch)

            valid_triples = [
                (chunk, dense_emb, sparse_emb)
                for chunk, dense_emb, sparse_emb in zip(
                    chunks, dense_embeddings, sparse_embeddings
                )
                if dense_emb is not None and sparse_emb is not None
            ]

            if not valid_triples:
                logger.warning(
                    "[%s] No valid chunks for %s, skipping.",
                    collection_name,
                    file_path.name,
                )
                continue

            valid_chunks = [item[0] for item in valid_triples]
            valid_dense = [item[1] for item in valid_triples]
            valid_sparse = [item[2] for item in valid_triples]

            base_payload = {
                "name": file_path.name,
                "file_path": str(file_path),
                "file_format": file_format,
                "creation_date": str(metadata.get("creation_date")),
                "modification_date": str(metadata.get("modification_date")),
            }

            logger.info(
                "[%s] Preparing to upsert %d chunks from %s",
                collection_name,
                len(valid_chunks),
                file_path,
            )

            await upsert_data(
                client=client_config.qdrant_client,
                collection_name=collection_name,
                embedding_model_config=embedding_model_config,
                dense_embeddings=valid_dense,
                bm25_embeddings=valid_sparse,
                late_interaction_embeddings=None,
                base_payload=base_payload,
                chunks=valid_chunks,
            )

            logger.info(
                "[%s] Successfully ingested %d points from %s",
                collection_name,
                len(valid_chunks),
                file_path.name,
            )

        except Exception as e:
            logger.error(
                "[%s] Failed to process file %s: %s",
                collection_name,
                file_path.name,
                e,
                exc_info=True,
            )
            continue  # Move to the next file


async def main() -> None:
    """
    Orchestrate the ingestion process with adaptive concurrency control.

    Scans the documents folder, organizes files by collection,
    performs memory calibration on up to three collections using tracemalloc,
    then processes the remaining collections in parallel with a dynamically
    computed concurrency limit.
    """
    # Initialize configurations and models inside main to avoid module‑level side effects
    app_config = AppConfig()
    client_config = ClientsConfig()
    db_config = DBConfig()
    embedding_model_config = EmbeddingModelsConfig()
    nlp_config = NLPConfig()

    folder_path = Path(__file__).parent.parent / "documents"
    logger.info("Scanning folder: %s", folder_path)

    files_by_collection: dict[str, list[Path]] = defaultdict(list)
    for path in folder_path.rglob("*"):
        if path.is_file():
            try:
                rel_path = path.relative_to(folder_path)
                collection_name = rel_path.parts[0]
                files_by_collection[collection_name].append(path)
            except IndexError:
                logger.warning("File outside any subfolder, skipping: %s", path)
                continue

    logger.info(
        "Found %d collections: %s",
        len(files_by_collection),
        list(files_by_collection.keys()),
    )

    if not files_by_collection:
        logger.info("No collections found. Exiting.")
        return

    # Helper to compute total size of files in a collection (MB)
    def get_total_size(file_paths: list[Path]) -> float:
        total = 0.0
        for f in file_paths:
            try:
                total += f.stat().st_size
            except OSError:
                continue
        return total / (1024 ** 2)

    all_items = sorted(
        files_by_collection.items(),
        key=lambda x: get_total_size(x[1])
    )
    n = len(all_items)

    if n <= 3:
        calibration_items = all_items
        remaining_items = []
    else:
        mid = n // 2
        calibration_items = all_items[mid - 1 : mid + 2]
        remaining_items = all_items[:mid - 1] + all_items[mid + 2 :]

    memory_stats = []

    # Start tracemalloc for accurate memory measurement
    tracemalloc.start()

    # Create a single shared aiohttp session for the entire application
    async with aiohttp.ClientSession() as session:
        # Phase 1: sequential calibration on up to three collections
        for collection, file_paths in calibration_items:
            logger.info("Calibrating on collection '%s'...", collection)

            # Take memory snapshot before processing
            snapshot1 = tracemalloc.take_snapshot()

            await process_collection(
                session,
                collection,
                file_paths,
                app_config,
                client_config,
                db_config,
                embedding_model_config,
                nlp_config,
            )

            # Force garbage collection to clean up cycles
            gc.collect()

            # Take snapshot after processing
            snapshot2 = tracemalloc.take_snapshot()

            # Compute memory allocated during this collection (positive differences only)
            stats = snapshot2.compare_to(snapshot1, "lineno")
            allocated_bytes = sum(stat.size_diff for stat in stats if stat.size_diff > 0)
            allocated_mb = allocated_bytes / (1024 ** 2)
            # Floor at 100 MB to avoid unrealistically low numbers
            used_mb = max(allocated_mb, 100.0)
            memory_stats.append(used_mb)

            logger.info(
                "Calibration: '%s' allocated ~%.1f MB in Python objects",
                collection,
                used_mb,
            )

        tracemalloc.stop()

        if memory_stats:
            if len(memory_stats) >= 2:
                mean_mem = statistics.mean(memory_stats)
                stdev_mem = statistics.stdev(memory_stats)
                # Use mean + one standard deviation as a robust estimate, then add 50% margin
                estimated = (mean_mem + stdev_mem) * 1.5
                logger.info(
                    "Memory estimate: mean=%.1f MB, stdev=%.1f MB",
                    mean_mem,
                    stdev_mem,
                )
            else:
                estimated = max(memory_stats) * 1.5
                logger.info("Memory estimate: max=%.1f MB", max(memory_stats))
        else:
            estimated = 2000.0  # Conservative fallback (2 GB per document)

        limit = compute_semaphore_limit(
            target_fraction=0.5,
            estimated_mb_per_doc=estimated,
            min_concurrency=1,
        )
        logger.info(
            "Computed concurrency limit: %d (based on ~%.0f MB per document)",
            limit,
            estimated,
        )

        if not remaining_items:
            logger.info("All collections processed during calibration phase.")
            return

        # Phase 2: parallel processing of remaining collections
        sem = asyncio.Semaphore(limit)

        async def bounded_process(collection: str, file_paths: list[Path]) -> None:
            async with sem:
                # Wait for memory to be available, then add a small jitter
                # to avoid thundering herd when multiple tasks wake up simultaneously.
                await wait_for_memory(logger, max_ram_fraction=0.5)
                await asyncio.sleep(random.uniform(0.1, 0.5))
                await process_collection(
                    session,
                    collection,
                    file_paths,
                    app_config,
                    client_config,
                    db_config,
                    embedding_model_config,
                    nlp_config,
                )

        tasks = [bounded_process(c, f) for c, f in remaining_items]
        await asyncio.gather(*tasks)

    logger.info("All collections processed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
