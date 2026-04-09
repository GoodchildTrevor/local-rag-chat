"""Answer cache: Redis session store + Qdrant upsert via qdrant-ingester."""
from __future__ import annotations

import json
import uuid
from logging import Logger
from datetime import datetime
from uuid import UUID

from config.settings import ClientsConfig, AppConfig
from databases.ingestion.client import IngesterClient


class AnswerCash:
    def __init__(
        self,
        logger: Logger,
        clients_config: ClientsConfig,
        app_config: AppConfig,
        collection_name: str,
        session_id: str,
        timeout_minutes: int,
    ) -> None:
        self.logger = logger
        self.client = clients_config.qdrant_client
        self.redis = clients_config.redis_client
        self.collection_name = collection_name
        self.session_id = session_id
        self.timeout_minutes = timeout_minutes
        self._ingester = IngesterClient(base_url=app_config.qdrant_ingester_url)

    def _session_key(self, session_id: str) -> str:
        return f"chat_session:{session_id}"

    async def add(
        self,
        user_id: str,
        question_id: UUID | None,
        msg: str,
        answer: str,
        display_docs: list,
        rating: float | None,
    ) -> None:
        key = self._session_key(self.session_id)
        now = datetime.utcnow().isoformat()
        entry = {
            "user_id": user_id,
            "question_id": str(question_id) if question_id else None,
            "msg": msg,
            "answer": answer,
            "display_docs": display_docs,
            "rating": rating,
            "timestamp": now,
        }
        self.logger.info("[ADD] Adding entry to %s: rating=%s, question_id=%s", key, rating, question_id)
        await self.redis.rpush(key, json.dumps(entry))
        await self.redis.expire(key, self.timeout_minutes * 60)

    async def flush(self, immidiate: bool = False) -> None:
        self.logger.info("[FLUSH] Starting flush...")
        keys = await self.redis.keys("chat_session:*")

        for key in keys:
            if not immidiate:
                ttl = await self.redis.ttl(key)
                if ttl > 0:
                    continue

            self.logger.info("[FLUSH] Processing session key: %s", key)
            raw_messages = await self.redis.lrange(key, 0, -1)
            if not raw_messages:
                continue

            latest_by_question: dict = {}
            for raw in raw_messages:
                msg = json.loads(raw)
                qid = msg["question_id"]
                ts = datetime.fromisoformat(msg["timestamp"])
                if qid not in latest_by_question or ts > datetime.fromisoformat(latest_by_question[qid]["timestamp"]):
                    latest_by_question[qid] = msg

            qas = [
                {
                    "question_id": m["question_id"],
                    "answer": m["answer"],
                    "display_docs": m["display_docs"],
                    "rating": int(m["rating"]),
                    "question": m["msg"],
                }
                for m in latest_by_question.values()
            ]

            if qas:
                self.logger.info("[FLUSH] Saving %d QAs...", len(qas))
                await self.save_answer(qas)

            await self.redis.delete(key)
            self.logger.info("[FLUSH] Deleted session key: %s", key)

    async def save_answer(self, qas: list[dict]) -> None:
        """
        Upsert Q&A pairs into the cache Qdrant collection via qdrant-ingester.
        """
        for qa in qas:
            question = qa["question"]
            question_id = qa["question_id"]
            rating = qa["rating"]
            display_docs = qa["display_docs"]

            try:
                if question_id:
                    point = self.client.retrieve(
                        collection_name=self.collection_name,
                        ids=[question_id],
                        with_payload=True,
                    )
                    current = point[0].payload
                    count = current.get("rating_count", 0)
                    avg = current.get("rating", 0.0)
                    new_rating = ((avg * count) + rating) / (count + 1)
                    payload = {
                        "rating": new_rating,
                        "rating_count": count + 1,
                        "question": question_id,
                        "display_docs": display_docs,
                    }
                    self.logger.info("[SAVE] Updating QA %s → avg_rating=%.2f", question_id, new_rating)
                else:
                    group_id = str(uuid.uuid4())
                    payload = {
                        "rating": rating,
                        "rating_count": 1,
                        "question": question,
                        "display_docs": display_docs,
                        "group_id": group_id,
                    }
                    self.logger.info("[SAVE] Creating new QA %s", group_id)

            except Exception as exc:
                self.logger.exception("[SAVE] Failed to prepare QA: %s", exc)
                continue

            try:
                await self._ingester.ingest_text(
                    collection=self.collection_name,
                    text=question,
                    payload=payload,
                )
            except Exception as exc:
                self.logger.exception("[SAVE] Upsert failed: %s", exc)
                continue

            self.logger.info("[SAVE] Upsert completed for question_id=%s", question_id)
