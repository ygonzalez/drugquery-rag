"""
Vector store operations using Weaviate.

This module handles:
- Weaviate connection management
- Schema creation and management
- Document embedding and indexing
- Hybrid search (semantic + keyword)

Usage:
    from drugquery.vectorstore import DrugVectorStore

    store = DrugVectorStore()
    store.index_chunks(chunks)
    results = store.hybrid_search("What are the side effects?")
"""

from drugquery.vectorstore.client import (
    DrugVectorStore,
    get_weaviate_client,
)
from drugquery.vectorstore.schema import (
    create_schema,
    delete_schema,
    get_collection_stats,
    COLLECTION_NAME,
)
from drugquery.vectorstore.embeddings import (
    DrugEmbedder,
    get_embedder,
)

__all__ = [
    "DrugVectorStore",
    "get_weaviate_client",
    "create_schema",
    "delete_schema",
    "get_collection_stats",
    "COLLECTION_NAME",
    "DrugEmbedder",
    "get_embedder",
]