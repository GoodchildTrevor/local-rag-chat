# Chunk info
CHUNK_SIZE = 384  # Standard value maybe should replace
OVERLAP = 1 # one sentence
# Work with DB
SCROLL_LIMIT = 1024
BATCH_SIZE = 64
# PDF reader
PDF_SIZE_LIMIT = 50
DPI = 400
# Embeddings
DENSE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_VECTOR_CONFIG = "all-MiniLM-L6-v2"
SPARSE_EMBEDDING_MODEL = "Qdrant/bm25"
SPARSE_VECTOR_CONFIG = "bm25"
LATE_EMBEDDING_MODEL = "colbert-ir/colbertv2.0"
LATE_VECTOR_CONFIG = "colbertv2.0"
# Doc info
FILE_FORMATS = [
    ".pdf",
    "doc",
    "docx",
]
