from dotenv import load_dotenv
import logging
from more_itertools import chunked
import os
from pathlib import Path

from fastembed import (
    TextEmbedding,
    LateInteractionTextEmbedding,
    SparseTextEmbedding,
)

from database.document_upserting.data_processing import (
    chunker,
    extract_text_metadata,
)
from database.document_upserting.data_loader import upsert_data, get_new_file_paths
from config.database import (
    BATCH_SIZE,
    DENSE_EMBEDDING_MODEL,
    SPARSE_EMBEDDING_MODEL,
    LATE_EMBEDDING_MODEL,
    CHUNK_SIZE,
    OVERLAP,
    FILE_FORMATS,
    SCROLL_LIMIT,
)
from config.settings import get_settings

load_dotenv()

collection = os.getenv("RAG_DOC_COLLECTION")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("../qdrant_ingest.log", mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
settings = get_settings()
client = settings.client

dense_embedding_model = TextEmbedding(DENSE_EMBEDDING_MODEL)
bm25_embedding_model = SparseTextEmbedding(SPARSE_EMBEDDING_MODEL)
late_interaction_embedding_model = LateInteractionTextEmbedding(LATE_EMBEDDING_MODEL)

folder_path = Path(__file__).parent.parent / 'documents'
files = dict()
logger.info(f"Scanning folder: {folder_path}")

target_paths = {path for path in folder_path.rglob('*') if path.is_file()}
paths_for_etl = get_new_file_paths(
    client=client,
    collection=collection,
    target_paths=target_paths,
    payload_key="file_path",
    scroll_limit=SCROLL_LIMIT,
)
logger.info(f"{len(paths_for_etl)} documents for upsert")

for file_path in paths_for_etl:
    file_format = file_path.suffix.lower()
    if file_format in FILE_FORMATS:
        logger.info(f"Processing file: {file_path.name}")
        text, metadata = extract_text_metadata(logger, file_path, file_format)

        chunks = chunker(text, CHUNK_SIZE, OVERLAP)
        raw_texts = [c['raw'] for c in chunks]
        lemma_texts = [c['lemmas'] for c in chunks]
        logger.info(f"{file_path.name}: {len(chunks)} chunks generated")
        # Initializing of empty lists
        dense_embeddings, bm25_embeddings, late_interaction_embeddings = [], [], []
        for batch in chunked(lemma_texts, BATCH_SIZE):
            dense_embeddings.extend(dense_embedding_model.embed(batch))
            bm25_embeddings.extend(bm25_embedding_model.embed(batch))
            late_interaction_embeddings.extend(late_interaction_embedding_model.embed(batch))

        upsert_data(
            client=client,
            collection_name=collection,
            dense_embeddings=dense_embeddings,
            bm25_embeddings=bm25_embeddings,
            late_interaction_embeddings=late_interaction_embeddings,
            name=file_path.name,
            documents=raw_texts,
            metadata=metadata,
            file_path=str(file_path)
        )

        logger.info(f"✅ Successfully ingested {len(chunks)} points from {file_path.name}")
    else:
        logger.debug(f"Missing unknown file extension: {file_path}")
