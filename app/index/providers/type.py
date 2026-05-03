from enum import Enum

class IndexType(Enum):
    SUMMARY = 1,
    VECTOR = 2,
    KEYWORD = 3,
    HYBRID = 4,

class QueryEngineType(Enum):
    CUSTOM = "custom"
    AUTO = "auto"
    BM25 = "bm25"
    RERANK = "rerank"