import os
from dotenv import load_dotenv

from functools import lru_cache
from pydantic import BaseModel
from pydantic_settings import BaseSettings

import pymorphy2
from pymorphy2 import MorphAnalyzer
from stop_words import get_stop_words

from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient

from config.database import (
    DENSE_EMBEDDING_MODEL,
    SPARSE_EMBEDDING_MODEL,
    LATE_EMBEDDING_MODEL,
    DENSE_VECTOR_CONFIG,
    SPARSE_VECTOR_CONFIG,
    LATE_VECTOR_CONFIG,
)
from config.searching import (
    TOP_K,
    RELEVANCY_THRESHOLD,
    FAITHFULNESS_THRESHOLD,
    DENSE_LIMIT,
    SPARSE_LIMIT,
    DENSE_THRESHOLD,
    SPARSE_THRESHOLD,
    THRESHOLD
)

load_dotenv()

RU_STOPWORDS = set(get_stop_words("ru"))
morph = pymorphy2.MorphAnalyzer()

HOST = os.getenv("HOST")
DB_PORT = os.getenv("PORT")
APP_PORT = os.getenv("APP_PORT")


class Settings(BaseSettings):
    app_port: int = APP_PORT
    client: QdrantClient = QdrantClient(HOST, port=DB_PORT)
    model_path: str
    top_k: int = TOP_K
    relevancy_threshold: float = RELEVANCY_THRESHOLD
    faithfulness_threshold: float = FAITHFULNESS_THRESHOLD
    stopwords: set = RU_STOPWORDS
    morph: MorphAnalyzer = morph
    dense_embedding_model: TextEmbedding = TextEmbedding(DENSE_EMBEDDING_MODEL)
    bm25_embedding_model: SparseTextEmbedding = SparseTextEmbedding(SPARSE_EMBEDDING_MODEL)
    late_interaction_embedding_model: LateInteractionTextEmbedding = LateInteractionTextEmbedding(LATE_EMBEDDING_MODEL)
    dense_vector_config: str = DENSE_VECTOR_CONFIG
    sparse_vector_config: str = SPARSE_VECTOR_CONFIG
    late_vector_config: str = LATE_VECTOR_CONFIG
    dense_limit: int = DENSE_LIMIT
    sparse_limit: int = SPARSE_LIMIT
    dense_threshold: float = DENSE_THRESHOLD
    sparse_threshold: float = SPARSE_THRESHOLD
    threshold: float = THRESHOLD

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class ChatRequest(BaseModel):
    question: str
    history: list[tuple[str, str]]


@lru_cache
def get_settings() -> Settings:
    """
    Returns the application settings.
    :return: Application configuration object.
    """
    return Settings()

