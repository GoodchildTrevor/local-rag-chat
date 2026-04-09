#!/bin/sh
set -e

echo "🔧 Initializing RAG app..."

# ---------------------------------------------------------------------------
# 1. Wait for Qdrant to be ready
# ---------------------------------------------------------------------------
echo "⏳ Waiting for Qdrant at ${HOST}:${DB_PORT}..."
until curl -sf "http://${HOST}:${DB_PORT}/readyz" > /dev/null; do
    sleep 2
done
echo "✅ Qdrant is ready."

# ---------------------------------------------------------------------------
# 2. Wait for qdrant-ingester to be ready
# ---------------------------------------------------------------------------
INGESTER_HOST=$(echo "${QDRANT_INGESTER_URL}" | sed 's|http://||' | cut -d: -f1)
INGESTER_PORT=$(echo "${QDRANT_INGESTER_URL}" | sed 's|http://||' | cut -d: -f2)
echo "⏳ Waiting for qdrant-ingester at ${INGESTER_HOST}:${INGESTER_PORT}..."
until curl -sf "http://${INGESTER_HOST}:${INGESTER_PORT}/health" > /dev/null; do
    sleep 3
done
echo "✅ qdrant-ingester is ready."

# ---------------------------------------------------------------------------
# 3. Wait for qdrant-searcher to be ready (loads ML models — may take ~60s)
# ---------------------------------------------------------------------------
SEARCHER_HOST=$(echo "${QDRANT_SEARCHER_URL}" | sed 's|http://||' | cut -d: -f1)
SEARCHER_PORT=$(echo "${QDRANT_SEARCHER_URL}" | sed 's|http://||' | cut -d: -f2)
echo "⏳ Waiting for qdrant-searcher at ${SEARCHER_HOST}:${SEARCHER_PORT} (ML models loading)..."
until curl -sf "http://${SEARCHER_HOST}:${SEARCHER_PORT}/health" > /dev/null; do
    sleep 5
done
echo "✅ qdrant-searcher is ready."

# ---------------------------------------------------------------------------
# 4. Create Qdrant collections (idempotent — skips if already exist)
# ---------------------------------------------------------------------------
echo "🗄️  Creating collections if not exist..."
python -m databases.collection_creator.collection_creator
echo "✅ Collections ready."

# ---------------------------------------------------------------------------
# 5. Trigger document ingestion via qdrant-ingester
#    IngesterClient.ingest_folder() is called inside main.py on startup,
#    so no separate ETL step is needed here.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 6. Start the app (NiceGUI requires __mp_main__ — must run via python, not uvicorn)
# ---------------------------------------------------------------------------
echo "🚀 Starting local-rag-chat..."
exec python main.py
