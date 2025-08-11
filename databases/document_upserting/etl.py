import logging
from more_itertools import chunked
from pathlib import Path

from databases.document_upserting.data_processing import (
    chunker,
    extract_text_metadata,
)
from databases.document_upserting.data_loader import upsert_data, get_new_file_paths
from config.settings import AppConfig, EmbeddingModelsConfig, ClientsConfig, DBConfig, NLPConfig

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("../qdrant_ingest.log", mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
app_config = AppConfig()
client_config = ClientsConfig()
db_config = DBConfig()
embeding_model_config = EmbeddingModelsConfig()
nlp_config = NLPConfig()

dense_embedding_model = embeding_model_config.dense
bm25_embedding_model = embeding_model_config.sparse
late_interaction_embedding_model = embeding_model_config.late

folder_path = Path(__file__).parent.parent / 'documents'
files = dict()
logger.info(f"Scanning folder: {folder_path}")

target_paths = {path for path in folder_path.rglob('*') if path.is_file()}
paths_for_etl = get_new_file_paths(
    client=client_config.qdrant_client,
    collection=app_config.rag_collection,
    target_paths=target_paths,
    payload_key="file_path",
    scroll_limit=db_config.scroll_limit,
)
logger.info(f"{len(paths_for_etl)} documents for upsert")

for file_path in paths_for_etl:
    file_format = file_path.suffix.lower()
    if file_format in db_config.file_format:
        logger.info(f"Processing file: {file_path.name}")
        text, metadata = extract_text_metadata(logger, file_path, file_format)

        chunks = chunker(nlp_config, text, db_config.chunk_size, db_config.overlap)
        raw_texts = [c['raw'] for c in chunks]
        lemma_texts = [c['lemmas'] for c in chunks]
        logger.info(f"{file_path.name}: {len(chunks)} chunks generated")
        # Initializing of empty lists
        dense_embeddings, bm25_embeddings, late_interaction_embeddings = [], [], []
        for batch in chunked(lemma_texts, db_config.batch_size):
            dense_embeddings.extend(dense_embedding_model.embed(batch))
            bm25_embeddings.extend(bm25_embedding_model.embed(batch))
            late_interaction_embeddings.extend(late_interaction_embedding_model.embed(batch))

        payload = {
            "name": file_path.name,
            "metadata": metadata,
            "file_path": str(file_path),
        },

        total_points = len(raw_texts)
        logger.info(f"Preparing to upsert {total_points} chunks from {file_path}")

        upsert_data(
            client=client_config.qdrant_client,
            collection_name=app_config.rag_collection,
            dense_embeddings=dense_embeddings,
            bm25_embeddings=bm25_embeddings,
            late_interaction_embeddings=late_interaction_embeddings,
            documents=raw_texts,
        )

        logger.info(f"✅ Successfully ingested {len(chunks)} points from {file_path.name}")
    else:
        logger.debug(f"Missing unknown file extension: {str(file_path)}")
