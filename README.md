# local-rag-chat

A fully **offline, self-hosted RAG chat** built on FastAPI, NiceGUI, Qdrant, and Ollama.
Documents are parsed, chunked, embedded, and indexed by dedicated microservices — the main app only handles conversation and LLM inference.

---

## ✨ Features

- 💬 Context-aware answers from a local Ollama LLM (llama.cpp-compatible)
- 🔍 Hybrid search — dense (`bge-m3`) + sparse (`BM25`) + late-interaction (`ColBERT`) via **qdrant-searcher**
- 📄 Document ingestion pipeline — PDF, DOCX, XLSX with OCR support — via **document-chunker** + **qdrant-ingester**
- 🧠 Answer cache — best Q&A pairs stored in Qdrant and surfaced on repeat queries
- 🖥️ Web UI via **[NiceGUI](https://nicegui.io/)**
- 🐳 Fully containerised — single `docker compose up` starts everything

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        docker network                        │
│                                                              │
│   ┌──────────┐   POST /vector_search   ┌─────────────────┐  │
│   │          │ ──────────────────────► │ qdrant-searcher │  │
│   │          │                         │  :8033          │  │
│   │   app    │   POST /ingest          └────────┬────────┘  │
│   │  :8000   │ ──────────────────────►          │           │
│   │ NiceGUI  │   POST /ingest_text   ┌──────────▼────────┐  │
│   │ FastAPI  │ ──────────────────────► qdrant-ingester   │  │
│   │          │                       │  :8002            │  │
│   └──────────┘                       └──────────┬────────┘  │
│                                                  │ POST /chunk│
│                                       ┌──────────▼────────┐  │
│                                       │ document-chunker  │  │
│                                       │  :8001            │  │
│                                       └───────────────────┘  │
│                                                              │
│   ┌──────────────┐          ┌───────────┐                   │
│   │    Qdrant    │          │   Redis   │                   │
│   │    :6333     │          │   :6379   │                   │
│   └──────────────┘          └───────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

| Service | Role |
|---|---|
| `app` | NiceGUI chat UI + FastAPI, LLM inference via Ollama |
| `qdrant-searcher` | Hybrid vector search (dense + sparse + ColBERT) |
| `qdrant-ingester` | Embeds chunks and upserts into Qdrant |
| `document-chunker` | Parses PDF/DOCX/XLSX, lemmatises, splits into chunks |
| `qdrant` | Vector database |
| `redis` | Session cache for conversation history |

---

## 🚀 Getting Started

### 1. Prerequisites

- [Docker + Docker Compose](https://docs.docker.com/engine/install/)
- [Ollama](https://ollama.com/) running locally with your chosen model pulled:
  ```bash
  ollama pull bge-m3       # dense embeddings
  ollama pull llama3       # or any chat model you prefer
  ```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
# App
APP_PORT=8000

# Qdrant
DB_PORT=6333
RAG_DOC_COLLECTION=rag_documents
CASH_COLLECTION=rag_cache

# Redis
REDIS_PORT=6379

# Session
SESSION_TIMEOUT_MINUTES=30

# Chunking
CHUNK_SIZE=512
OVERLAP=1
```

The microservice URLs (`QDRANT_SEARCHER_URL`, `QDRANT_INGESTER_URL`, `DOCUMENT_CHUNKER_URL`) are pre-filled in `.env.example` for the Docker network and do not need to be changed.

### 3. Start all services

```bash
docker compose up -d
```

> **Note:** `qdrant-searcher` loads ML models on startup — allow ~60 seconds before the first search request.

### 4. Add documents

Drop your files into `databases/documents/<collection_name>/`.
The folder name becomes the Qdrant collection name.

Then trigger ingestion (the app does this automatically on startup via `IngesterClient.ingest_folder`, or you can call the API directly):

```bash
curl -X POST http://localhost:8002/ingest \
  -H "Content-Type: application/json" \
  -d '{"collection": "rag_documents", "file_path": "/app/databases/documents/rag_documents/myfile.pdf"}'
```

### 5. Open the chat

```
http://localhost:8000
```

---

## 📂 Project Structure

```
local-rag-chat/
├── chat/
│   ├── interface/              # NiceGUI pages, tabs, chat widgets
│   └── backend/
│       └── dialogue.py         # Query normalisation + delegates to qdrant-searcher
├── config/
│   ├── settings.py             # AppConfig, ClientsConfig, NLPConfig
│   └── consts/                 # Search thresholds, prompts, tab config
├── databases/
│   ├── cashing/
│   │   └── cashing.py          # Redis session store + Q&A upsert via qdrant-ingester
│   ├── ingestion/
│   │   └── client.py           # HTTP client for qdrant-ingester
│   ├── searcher/
│   │   └── searcher_client.py  # HTTP client for qdrant-searcher
│   └── documents/              # Raw documents to index (gitignored)
├── llm/
│   ├── ollama_configs.py       # LLM model definitions
│   └── ollama_inference.py     # ask_llm() streaming wrapper
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── main.py                     # App entry point
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `APP_PORT` | `8000` | Port for the NiceGUI / FastAPI app |
| `DB_PORT` | `6333` | Qdrant port |
| `REDIS_PORT` | `6379` | Redis port |
| `RAG_DOC_COLLECTION` | — | Main Qdrant collection name |
| `CASH_COLLECTION` | — | Answer-cache collection name |
| `SESSION_TIMEOUT_MINUTES` | `30` | Redis session TTL |
| `CHUNK_SIZE` | `512` | Token chunk size (passed to document-chunker) |
| `OVERLAP` | `1` | Chunk overlap (sentences) |
| `DENSE_MODEL_NAME` | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | Dense embedding model for ingester |
| `SPARSE_MODEL_NAME` | `Qdrant/bm25` | Sparse model for ingester |
| `QDRANT_SEARCHER_URL` | `http://qdrant-searcher:8033` | Pre-filled for Docker network |
| `QDRANT_INGESTER_URL` | `http://qdrant-ingester:8002` | Pre-filled for Docker network |
| `DOCUMENT_CHUNKER_URL` | `http://document-chunker:8001/chunk` | Pre-filled for Docker network |

---

## 🔌 Related Services

| Repository | Description |
|---|---|
| [qdrant-searcher](https://github.com/GoodchildTrevor/qdrant-searcher) | Standalone hybrid search API (dense + sparse + ColBERT) |
| `qdrant-ingester` | Embedding + upsert service *(coming soon)* |
| `document-chunker` | Parser + chunker service *(coming soon)* |
