"""HTTP client for qdrant-searcher microservice."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class HybridHit:
    """Mirror of qdrant-searcher's HybridHit schema."""
    id: str
    score: float
    source: str
    payload: dict


class SearcherClient:
    """
    Async HTTP client for qdrant-searcher  POST /vector_search.

    Usage::

        client = SearcherClient(base_url="http://qdrant-searcher:8033")
        hits = await client.search(
            text="запрос пользователя",
            collection_name="rag_documents",
            method="hybrid",
            top_k=5,
        )
    """

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    async def search(
        self,
        text: str,
        collection_name: str,
        method: str = "hybrid",
        top_k: int = 5,
    ) -> list[HybridHit]:
        payload = {
            "text": text,
            "method": method,
            "collection_name": collection_name,
            "top_k": top_k,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as http:
            response = await http.post(f"{self._base}/vector_search", json=payload)
        response.raise_for_status()
        data = response.json()
        hits = [
            HybridHit(
                id=str(d["id"]),
                score=d["score"],
                source=d["source"],
                payload=d["payload"],
            )
            for d in data.get("documents", [])
        ]
        logger.info(
            "qdrant-searcher: collection=%s method=%s hits=%d",
            collection_name, method, len(hits),
        )
        return hits

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as http:
                r = await http.get(f"{self._base}/health")
            return r.status_code == 200
        except Exception:
            return False
