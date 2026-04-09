import os
from dotenv import load_dotenv

from pydantic import BaseModel, field as pydantic_field
from pydantic_settings import BaseSettings

import pymorphy3
from pymorphy3 import MorphAnalyzer
from stop_words import get_stop_words
import tiktoken

from llama_index.llms.ollama import Ollama
from qdrant_client import QdrantClient
from redis.asyncio import Redis

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
)

load_dotenv()

# Singletons initialised once at import time
RU_STOPWORDS = set(get_stop_words("ru"))
morph = pymorphy3.MorphAnalyzer()
tokenizer = tiktoken.get_encoding("cl100k_base")


class AppConfig(BaseSettings):
    app_port: int = pydantic_field(description="HTTP port the app listens on")
    timeout: int = pydantic_field(description="Session timeout in minutes")
    rag_collection: str = pydantic_field(description="Qdrant collection for RAG docs")
    cash_collection: str = pydantic_field(description="Qdrant collection for cache")
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

    # External microservice URLs
    qdrant_searcher_url: str = pydantic_field(
        description="Base URL of qdrant-searcher (e.g. http://qdrant-searcher:8033)"
    )
    document_chunker_url: str = pydantic_field(
        description="URL of document-chunker /chunk endpoint"
    )
    qdrant_ingester_url: str = pydantic_field(
        description="Base URL of qdrant-ingester (e.g. http://qdrant-ingester:8002)"
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


class ChatRequest(BaseModel):
    question: str
    history: list[tuple[str, str]]


class DBConfig(BaseSettings):
    rag_snapshot_dir: str = pydantic_field(
        description="Directory for Qdrant snapshots"
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


class ClientsConfig:
    def __init__(
        self,
        host: str = os.getenv("HOST"),
        db_port: str = os.getenv("DB_PORT"),
        redis_port: str = os.getenv("REDIS_PORT"),
    ):
        self.qdrant_url: str = f"http://{host}:{db_port}"
        self.qdrant_client: QdrantClient = QdrantClient(host, port=db_port)
        self.redis_client: Redis = Redis(
            host=host, port=redis_port, db=0, decode_responses=True
        )


class NLPConfig:
    """
    NLP tools for query-time normalisation (tokenise + lemmatise).
    Lives in app because Dialogue.processing_query() runs per-user-request.
    """
    def __init__(self):
        self.stopwords: set = RU_STOPWORDS
        self.morph: MorphAnalyzer = morph
        self.tokenizer: tiktoken.Encoding = tokenizer


class RAGTabConfig(TabConfig):
    prefix: str = "chat"
    header: str = "Чат-бот"
    system_prompt: str = RAG_SYSTEM_PROMPT
    markdown: str = ""
    llm: Ollama = chat_llm


class CodeAssistantTabConfig(TabConfig):
    prefix: str = "assistant"
    header: str = "Код ассистент"
    system_prompt: str = CODER_SYSTEM_PROMPT
    markdown: str = ""
    llm: Ollama = code_assistant_llm
