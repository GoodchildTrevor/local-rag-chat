import json
import uuid
from logging import Logger
from datetime import datetime
from uuid import UUID

from config.settings import ClientsConfig, EmbeddingModelsConfig
from databases.document_upserting.data_loader import upsert_data


class AnswerCash:
    def __init__(
        self,
        logger: Logger, 
        clients_config: ClientsConfig,
        embedding_model_config: EmbeddingModelsConfig,
        collection_name: str,
        session_id: str,
        timeout_minutes: int
    ):
        self.logger = logger
        self.client = clients_config.qdrant_client
        self.redis = clients_config.redis_client
        self.dense_embedding_model = embedding_model_config.dense
        self.collection_name = collection_name
        self.session_id = session_id
        self.timeout_minutes = timeout_minutes

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
    ):
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

        self.logger.info(f"[ADD] Adding entry to {key}: rating={rating}, question_id={question_id}")
        await self.redis.rpush(key, json.dumps(entry))
        await self.redis.expire(key, self.timeout_minutes * 60)

    async def flush(self, immidiate=False):
        self.logger.info("[FLUSH] Starting flush...")
        keys = await self.redis.keys("chat_session:*")

        for key in keys:
            if not immidiate:
                ttl = await self.redis.ttl(key)
                if ttl > 0:
                    continue

            self.logger.info(f"[FLUSH] Processing session key: {key}")
            raw_messages = await self.redis.lrange(key, 0, -1)
            if not raw_messages:
                self.logger.info(f"[FLUSH] No messages in {key}")
                continue

            latest_by_question = {}
            for raw in raw_messages:
                msg = json.loads(raw)
                qid = msg["question_id"]
                ts = datetime.fromisoformat(msg["timestamp"])
                if qid not in latest_by_question or ts > datetime.fromisoformat(latest_by_question[qid]["timestamp"]):
                    latest_by_question[qid] = msg

            self.logger.info(f"[FLUSH] Latest messages by question: {len(latest_by_question)}")

            qas = []
            for msg in latest_by_question.values():
                rating = msg.get("rating")

                qas.append({
                    "question_id": msg["question_id"],
                    "answer": msg["answer"],
                    "display_docs": msg["display_docs"],
                    "rating": int(rating),
                    "question": msg["msg"]
                })

            if qas:
                self.logger.info(f"[FLUSH] Saving {len(qas)} QAs...")
                await self.save_answer(qas)

            await self.redis.delete(key)
            self.logger.info(f"[FLUSH] Deleted session key: {key}")

    async def save_answer(self, qas: list[dict]):

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
                        with_payload=True
                    )
                    current_payload = point[0].payload
                    current_rating_count = current_payload.get("rating_count", 0)
                    current_avg_rating = current_payload.get("rating", 0.0)

                    new_rating = ((current_avg_rating * current_rating_count) + rating) / (current_rating_count + 1)
                    payload = {
                        "rating": new_rating,
                        "rating_count": current_rating_count + 1,
                        "question": question_id,
                        "display_docs": display_docs,
                    }
                    self.logger.info(f"[SAVE] Updating existing QA: {question_id} with new avg rating: {new_rating:.2f}")
                else:
                    group_id = str(uuid.uuid4())
                    payload = {
                        "rating": rating,
                        "rating_count": 1,
                        "question": question,
                        "display_docs": display_docs,
                        "group_id": group_id,
                    }
                    self.logger.info(f"[SAVE] Creating new QA: {group_id}")

            except Exception as e:
                self.logger.exception(f"[SAVE] Failed to prepare QA: {e}")

            try:
                try:
                    qa["question"]
                    dense_emb = list(self.dense_embedding_model.embed(qa["question"]))
                except Exception as e:
                    self.logger.exception(f"[FLUSH] Embedding error: {e}")
                    continue
                upsert_data(
                    client=self.client,
                    collection_name=self.collection_name,
                    dense_embeddings=dense_emb,
                    bm25_embeddings=None,
                    late_interaction_embeddings=None,
                    documents=[qa["answer"]],
                    payload=payload,
                )
            except Exception as e:
                self.logger.exception(f"[SAVE] Upsert failed: {e}")

            self.logger.info("[SAVE] Upsert completed successfully.")
