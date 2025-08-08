import os
from dotenv import load_dotenv

from pydantic import BaseModel
from pydantic_settings import BaseSettings

import pymorphy2
from pymorphy2 import MorphAnalyzer
from stop_words import get_stop_words

from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient
from redis.asyncio import Redis

from config.consts.database import (
    DENSE_EMBEDDING_MODEL,
    SPARSE_EMBEDDING_MODEL,
    LATE_EMBEDDING_MODEL,
    DENSE_VECTOR_CONFIG,
    SPARSE_VECTOR_CONFIG,
    LATE_VECTOR_CONFIG,
)
from config.consts.searching import (
    TOP_K,
    RELEVANCY_THRESHOLD,
    FAITHFULNESS_THRESHOLD,
    DENSE_LIMIT,
    SPARSE_LIMIT,
    DENSE_THRESHOLD,
    SPARSE_THRESHOLD,
    THRESHOLD,
    COSINE_SIMILARITY_THRESHOLD,
)

load_dotenv()

RU_STOPWORDS = set(get_stop_words("ru"))
morph = pymorphy2.MorphAnalyzer()

HOST = os.getenv("HOST")
DB_PORT = os.getenv("DB_PORT")
REDIS_PORT = os.getenv("REDIS_PORT")
APP_PORT = os.getenv("APP_PORT")
RAG_DOC_COLLECTION = os.getenv("RAG_DOC_COLLECTION")
CASH_COLLECTION = os.getenv("CASH_COLLECTION")
TIMEOUT = os.getenv('SESSION_TIMEOUT_MINUTES')


class AppConfig(BaseSettings):
    app_port: int = APP_PORT
    timeout: int = TIMEOUT
    rag_collection: str = RAG_DOC_COLLECTION
    cash_collection: str = CASH_COLLECTION
    top_k: int = TOP_K
    relevancy_threshold: float = RELEVANCY_THRESHOLD
    faithfulness_threshold: float = FAITHFULNESS_THRESHOLD
    dense_vector_config: str = DENSE_VECTOR_CONFIG
    sparse_vector_config: str = SPARSE_VECTOR_CONFIG
    late_vector_config: str = LATE_VECTOR_CONFIG
    dense_limit: int = DENSE_LIMIT
    sparse_limit: int = SPARSE_LIMIT
    dense_threshold: float = DENSE_THRESHOLD
    sparse_threshold: float = SPARSE_THRESHOLD
    threshold: float = THRESHOLD
    cosine_similarity_threshold: float = COSINE_SIMILARITY_THRESHOLD

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class ClientsConfig:
    def __init__(self, host=HOST, db_port=DB_PORT, redis_port=REDIS_PORT):
        self.qdrant_client: QdrantClient = QdrantClient(host, port=db_port)
        self.redis_client: Redis = Redis(host=host, port=redis_port, db=0, decode_responses=True)


class EmbeddingModelsConfig:
    def __init__(self):
        self.dense: TextEmbedding = TextEmbedding(DENSE_EMBEDDING_MODEL)
        self.sparse: SparseTextEmbedding = SparseTextEmbedding(SPARSE_EMBEDDING_MODEL)
        self.late: LateInteractionTextEmbedding = LateInteractionTextEmbedding(LATE_EMBEDDING_MODEL)


class NLPConfig:
    def __init__(self):
        self.stopwords: set = RU_STOPWORDS
        self.morph: MorphAnalyzer = morph


class ChatRequest(BaseModel):
    question: str
    history: list[tuple[str, str]]
