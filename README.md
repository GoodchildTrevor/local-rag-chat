# Offline RAG Pipeline with Qdrant and llama.cpp

This project implements a lightweight **Retrieval-Augmented Generation (RAG)** pipeline using **FastAPI**, **Qdrant**, and a local **llama.cpp** model. It runs fully offline — ideal for self-hosted or air-gapped environments.

## ✨ Features

- 📄 Local document ingestion with metadata support
- ⚡ OCR support for scanned documents via Tesseract
- 🔍 Hybrid search (semantic + full-text) powered by **Qdrant**
- 🧠 Context-aware answers from local **llama.cpp** model
- 🖼️ Web UI via **[NiceGUI](https://nicegui.io/)**

## 📦 Required Tools

### Qdrant
[Installation guide](https://qdrant.tech/documentation/guides/installation/)

### Ollama
[Installation guide](https://apxml.com/courses/getting-started-local-llms/chapter-4-running-first-local-llm/setting-up-ollama)

### Tesseract
[Installation guide](https://builtin.com/articles/python-tesseract)

## 🚀 Getting Started
Create a `.env` file in the root directory:

```
HOST=                 # your host
DB_PORT=              # port for Qdrant
RAG_DOC_COLLECTION=   # name of your collection
APP_PORT=             # port for FastAPI
```

```bash
# 1. Install dependencies
pip install -r requirements.txt
# 2. Create your document collection
python database/collection_creator/collection_creator.py
# 3. Upsert your documents
python database//document_upserting/etl.py
# 4. Run the API server
python main.py
# 5. Open the GUI
# http://localhost:8000
```

## 📂 Project Structure
```
RAG/
├── chat/
│   ├── interface/            # NiceGUI frontend and utilities
│   └── backend/              # Dialogue logic
├── config/                   # All constants for all components
├── database/                 # Database scripts
│   ├── collection_creator    # Create/recreate collection
│   ├── document_upserting    # Upsert documents
│   ├── documents/            # Raw documents to be indexed
│   └── search                # Seacrh engine
├── models/                   # Ollama config and inference scripts
└── main.py                   # Entry point
```
