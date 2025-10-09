# Chunk info
CHUNK_SIZE = 384  # Standard value maybe should replace
OVERLAP = 1 # one sentence
# Work with DB
SCROLL_LIMIT = 1024
BATCH_SIZE = 64
# PDF reader
PDF_SIZE_LIMIT = 50
DPI = 300
# Embeddings
DENSE_EMBEDDING_MODEL = "mxbai-embed-large"
SPARSE_EMBEDDING_MODEL = "Qdrant/bm25"
LATE_EMBEDDING_MODEL = "colbert-ir/colbertv2.0"

DENSE_VECTOR_CONFIG = "dense"
SPARSE_VECTOR_CONFIG = "sparse"
LATE_VECTOR_CONFIG = "late"

# Doc info
FILE_FORMATS = [
    ".pdf",
    "doc",
    "docx",
]
