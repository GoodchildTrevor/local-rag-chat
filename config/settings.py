import os
from dotenv import load_dotenv

from pydantic import BaseModel
from pydantic_settings import BaseSettings

import pymorphy3
from pymorphy3 import MorphAnalyzer
from stop_words import get_stop_words
import tiktoken

from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient
from redis.asyncio import Redis

from config.consts.database import (
    BATCH_SIZE,
    DENSE_EMBEDDING_MODEL,
    SPARSE_EMBEDDING_MODEL,
    LATE_EMBEDDING_MODEL,
    DENSE_VECTOR_CONFIG,
    SPARSE_VECTOR_CONFIG,
    LATE_VECTOR_CONFIG,
    CHUNK_SIZE,
    OVERLAP,
    FILE_FORMATS,
    SCROLL_LIMIT,
)
from config.consts.searching import (
    TOP_K,
    RELEVANCY_THRESHOLD,
    FAITHFULNESS_THRESHOLD,
    DENSE_LIMIT,
    SPARSE_LIMIT,
    LATE_LIMIT,
    DENSE_THRESHOLD,
    SPARSE_THRESHOLD,
    THRESHOLD,
    COSINE_SIMILARITY_THRESHOLD,
)
from config.consts.prompts import (
    CODER_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT,
)
from config.consts.tab_config import TabConfig
from llm.ollama_configs import (
    chat_llm,
    code_assistant_llm,
    FixedOllama
)

load_dotenv()

RU_STOPWORDS = set(get_stop_words("ru"))
morph = pymorphy3.MorphAnalyzer()
tokenizer = tiktoken.get_encoding("cl100k_base")

HOST = os.getenv("HOST")
DB_PORT = os.getenv("DB_PORT")
REDIS_PORT = os.getenv("REDIS_PORT")
APP_PORT = os.getenv("APP_PORT")
RAG_DOC_COLLECTION = os.getenv("RAG_DOC_COLLECTION")
CASH_COLLECTION = os.getenv("CASH_COLLECTION")
TIMEOUT = os.getenv('SESSION_TIMEOUT_MINUTES')
RAG_SNAPSHOT_DIR = os.getenv("RAG_SNAPSHOT_DIR")


class AppConfig(BaseSettings):
    app_port: int = APP_PORT
    timeout: int = TIMEOUT
    rag_collection: str = RAG_DOC_COLLECTION
    cash_collection: str = CASH_COLLECTION
    top_k: int = TOP_K
    relevancy_threshold: float = RELEVANCY_THRESHOLD
    faithfulness_threshold: float = FAITHFULNESS_THRESHOLD
    dense_limit: int = DENSE_LIMIT
    sparse_limit: int = SPARSE_LIMIT
    late_limit: int = LATE_LIMIT
    dense_threshold: float = DENSE_THRESHOLD
    sparse_threshold: float = SPARSE_THRESHOLD
    threshold: float = THRESHOLD
    cosine_similarity_threshold: float = COSINE_SIMILARITY_THRESHOLD

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class ChatRequest(BaseModel):
    question: str
    history: list[tuple[str, str]]


class DBConfig:
    def __init__(self):
        self.batch_size: int = BATCH_SIZE
        self.chunk_size: int = CHUNK_SIZE
        self.overlap: int = OVERLAP
        self.file_format: str = FILE_FORMATS
        self.scroll_limit: int = SCROLL_LIMIT
        self.rag_snapshot_dir: str = RAG_SNAPSHOT_DIR


class ClientsConfig:
    def __init__(self, host=HOST, db_port=DB_PORT, redis_port=REDIS_PORT):
        self.qdrant_url: str = f"http://{host}:{db_port}"
        self.qdrant_client: QdrantClient = QdrantClient(host, port=db_port)
        self.redis_client: Redis = Redis(host=host, port=redis_port, db=0, decode_responses=True)


class EmbeddingModelsConfig:
    def __init__(self):
        self.dense: TextEmbedding = TextEmbedding(DENSE_EMBEDDING_MODEL)
        self.sparse: SparseTextEmbedding = SparseTextEmbedding(SPARSE_EMBEDDING_MODEL)
        self.late: LateInteractionTextEmbedding = LateInteractionTextEmbedding(LATE_EMBEDDING_MODEL)
        self.dense_vector_config: str = DENSE_VECTOR_CONFIG
        self.sparse_vector_config: str = SPARSE_VECTOR_CONFIG
        self.late_vector_config: str = LATE_VECTOR_CONFIG


class NLPConfig:
    def __init__(self):
        self.stopwords: set = RU_STOPWORDS
        self.morph: MorphAnalyzer = morph
        self.tokenizer: tiktoken.Encoding = tokenizer 


class RAGTabConfig(TabConfig):
    prefix: str = "chat"
    header: str = "Чат-бот"
    system_prompt: str = RAG_SYSTEM_PROMPT
    markdown: str = ""
    llm: FixedOllama = chat_llm


class CodeAssistantTabConfig(TabConfig):
    prefix: str = "assistant"
    header: str = "Код ассистент"
    system_prompt: str = CODER_SYSTEM_PROMPT
    markdown: str = ""
    llm: FixedOllama = code_assistant_llm
