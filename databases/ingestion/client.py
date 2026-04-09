"""HTTP client for qdrant-ingester microservice."""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


class IngesterClient:
    """
    Thin async wrapper around qdrant-ingester REST API.

    Endpoints used:
      POST /ingest        — chunk + embed + upsert one file
      POST /ingest_text   — embed + upsert a raw text snippet (for cache Q&A)
      POST /sync          — diff current files vs Qdrant, delete orphans
    """

    def __init__(self, base_url: str, timeout: float = 600.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    # ------------------------------------------------------------------
    # File ingestion
    # ------------------------------------------------------------------

    async def ingest_file(
        self,
        collection: str,
        file_path: Path,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> dict:
        payload: dict = {"collection": collection, "file_path": str(file_path)}
        if chunk_size is not None:
            payload["chunk_size"] = chunk_size
        if overlap is not None:
            payload["overlap"] = overlap

        async with httpx.AsyncClient(timeout=self._timeout) as http:
            response = await http.post(f"{self._base}/ingest", json=payload)
        response.raise_for_status()
        result = response.json()
        logger.info(
            "Ingested '%s' into '%s': %d chunks upserted",
            file_path.name, collection, result.get("chunks_upserted", 0),
        )
        return result

    # ------------------------------------------------------------------
    # Raw text ingestion (cache Q&A)
    # ------------------------------------------------------------------

    async def ingest_text(
        self,
        collection: str,
        text: str,
        payload: dict | None = None,
    ) -> dict:
        """
        POST /ingest_text — embed a raw text string and upsert into Qdrant.
        Used by AnswerCash to persist Q&A pairs without going through chunker.
        """
        body = {"collection": collection, "text": text, "payload": payload or {}}
        async with httpx.AsyncClient(timeout=self._timeout) as http:
            response = await http.post(f"{self._base}/ingest_text", json=body)
        response.raise_for_status()
        result = response.json()
        logger.info("ingest_text into '%s': %s", collection, result)
        return result

    # ------------------------------------------------------------------
    # Collection sync
    # ------------------------------------------------------------------

    async def sync_collection(
        self,
        collection: str,
        current_file_paths: set[Path],
    ) -> tuple[list[Path], int]:
        payload = {
            "collection": collection,
            "current_file_paths": [str(p) for p in current_file_paths],
        }
        async with httpx.AsyncClient(timeout=self._timeout) as http:
            response = await http.post(f"{self._base}/sync", json=payload)
        response.raise_for_status()
        data = response.json()
        new_paths = [Path(p) for p in data.get("new_files", [])]
        deleted = data.get("deleted_chunks", 0)
        logger.info(
            "Sync '%s': %d new files, %d orphaned chunks deleted",
            collection, len(new_paths), deleted,
        )
        return new_paths, deleted

    # ------------------------------------------------------------------
    # Full folder ETL
    # ------------------------------------------------------------------

    async def ingest_folder(
        self,
        folder_path: Path,
        allowed_formats: set[str] | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> dict[str, int]:
        files_by_collection: dict[str, list[Path]] = defaultdict(list)
        for path in folder_path.rglob("*"):
            if not path.is_file():
                continue
            if allowed_formats and path.suffix.lower() not in allowed_formats:
                continue
            try:
                collection_name = path.relative_to(folder_path).parts[0]
            except IndexError:
                logger.warning("File outside any subfolder, skipping: %s", path)
                continue
            files_by_collection[collection_name].append(path)

        totals: dict[str, int] = {}
        for collection, file_paths in files_by_collection.items():
            new_paths, deleted = await self.sync_collection(collection, set(file_paths))
            upserted = 0
            for fp in new_paths:
                try:
                    result = await self.ingest_file(
                        collection=collection,
                        file_path=fp,
                        chunk_size=chunk_size,
                        overlap=overlap,
                    )
                    upserted += result.get("chunks_upserted", 0)
                except httpx.HTTPStatusError as e:
                    logger.error(
                        "Ingest failed for %s [HTTP %d]: %s",
                        fp.name, e.response.status_code, e.response.text[:200],
                    )
                except Exception as e:
                    logger.error("Ingest failed for %s: %s", fp.name, e)
            totals[collection] = upserted
            logger.info(
                "Collection '%s': %d new, %d upserted, %d orphans deleted",
                collection, len(new_paths), upserted, deleted,
            )
        return totals
