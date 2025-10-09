#!/bin/sh
set -e

echo "🔧 Initializing RAG..."

# 1. Creating collecction
python -m databases.collection_creator.collection_creator

echo "✅ Collection created..."

# 2. Upsert documents
python -m databases.document_upserting.etl

echo "✅ Documents upserted. Run chat"

# 3. Run chat
exec uvicorn main:app --host 0.0.0.0

