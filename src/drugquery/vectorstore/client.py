"""
Weaviate client for drug chunk storage and retrieval.

This module provides:
- Connection management (context manager pattern)
- Batch indexing with embeddings
- Hybrid search (vector + keyword)
- Filtering by drug name, section type, etc.

Usage:
    from drugquery.vectorstore.client import DrugVectorStore

    store = DrugVectorStore()

    # Index chunks
    store.index_chunks(chunks)

    # Search
    results = store.hybrid_search(
        query="What are the side effects of aspirin?",
        limit=5,
        section_filter="adverse_reactions",
    )
"""

from contextlib import contextmanager
from typing import Generator

import weaviate
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.classes.data import DataObject

from drugquery.config import settings
from drugquery.ingestion.chunk import DocumentChunk
from drugquery.vectorstore.schema import COLLECTION_NAME, create_schema, get_collection_stats
from drugquery.vectorstore.embeddings import DrugEmbedder, get_embedder
from drugquery.logging import get_logger

logger = get_logger(__name__, component="vectorstore")


@contextmanager
def get_weaviate_client() -> Generator[weaviate.WeaviateClient, None, None]:
    """
    Get a Weaviate client connection.

    Uses context manager pattern to ensure proper cleanup.

    Usage:
        with get_weaviate_client() as client:
            # Use client
            ...
        # Connection automatically closed
    """
    client = weaviate.connect_to_local(
        host=settings.weaviate_host,
        port=settings.weaviate_port,
    )

    try:
        logger.debug("weaviate_connected", host=settings.weaviate_host)
        yield client
    finally:
        client.close()
        logger.debug("weaviate_disconnected")


class DrugVectorStore:
    """
    Vector store for drug label chunks.

    Provides methods for:
    - Indexing chunks with embeddings
    - Hybrid search (semantic + keyword)
    - Filtering by metadata

    The store manages its own Weaviate connection and embedder.

    Example:
        store = DrugVectorStore()

        # Index all chunks
        store.index_chunks(chunks)

        # Search with filters
        results = store.hybrid_search(
            query="aspirin side effects",
            section_filter="adverse_reactions",
            limit=5,
        )

        # Clean up when done
        store.close()
    """

    def __init__(self, embedder: DrugEmbedder | None = None):
        """
        Initialize the vector store.

        Args:
            embedder: Optional embedder instance. If not provided,
                     uses the shared singleton embedder.
        """
        # Connect to Weaviate
        self.client = weaviate.connect_to_local(
            host=settings.weaviate_host,
            port=settings.weaviate_port,
        )

        # Use provided embedder or get shared instance
        self.embedder = embedder or get_embedder()

        # Ensure schema exists
        create_schema(self.client)

        # Get collection reference
        self.collection = self.client.collections.get(COLLECTION_NAME)

        logger.info("vectorstore_initialized", collection=COLLECTION_NAME)

    def close(self) -> None:
        """Close the Weaviate connection."""
        self.client.close()
        logger.debug("vectorstore_closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_stats(self) -> dict:
        """Get collection statistics."""
        return get_collection_stats(self.client)

    def index_chunks(
            self,
            chunks: list[DocumentChunk],
            batch_size: int = 100,
    ) -> int:
        """
        Index chunks with embeddings into Weaviate.

        This method:
        1. Extracts text content from chunks
        2. Generates embeddings using the embedder
        3. Batch inserts into Weaviate with vectors

        Args:
            chunks: List of DocumentChunk objects to index
            batch_size: Number of chunks to insert per batch

        Returns:
            Number of chunks successfully indexed
        """
        if not chunks:
            logger.warning("no_chunks_to_index")
            return 0

        logger.info("indexing_start", total_chunks=len(chunks))

        # Step 1: Generate embeddings for all chunks
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.embed_documents(texts, batch_size=batch_size)

        # Step 2: Batch insert into Weaviate
        indexed_count = 0
        failed_count = 0

        with self.collection.batch.dynamic() as batch:
            for chunk, embedding in zip(chunks, embeddings):
                try:
                    # Prepare properties
                    properties = {
                        "chunk_id": chunk.chunk_id,
                        "drug_name": chunk.drug_name,
                        "generic_name": chunk.generic_name or "",
                        "section_type": chunk.section_type,
                        "section_title": chunk.section_title,
                        "content": chunk.content,
                        "token_count": chunk.token_count,
                        "chunk_index": chunk.chunk_index,
                        "total_chunks": chunk.total_chunks,
                        "set_id": chunk.metadata.get("set_id", ""),
                        "manufacturer": chunk.metadata.get("manufacturer") or "",
                        "source_url": chunk.metadata.get("source_url", ""),
                    }

                    # Add to batch with vector
                    batch.add_object(
                        properties=properties,
                        vector=embedding.tolist(),
                    )
                    indexed_count += 1

                except Exception as e:
                    logger.warning(
                        "index_chunk_failed",
                        chunk_id=chunk.chunk_id,
                        error=str(e),
                    )
                    failed_count += 1

        logger.info(
            "indexing_complete",
            indexed=indexed_count,
            failed=failed_count,
        )

        return indexed_count

    def hybrid_search(
            self,
            query: str,
            limit: int = 10,
            alpha: float | None = None,
            section_filter: str | None = None,
            drug_filter: str | None = None,
    ) -> list[dict]:
        """
        Perform hybrid search combining vector and keyword search.

        Hybrid search gives the best of both worlds:
        - Vector search: Finds semantically similar content
        - Keyword search: Finds exact term matches

        The alpha parameter controls the balance:
        - alpha=1.0: Pure vector search
        - alpha=0.0: Pure keyword search
        - alpha=0.7: Mostly vector, some keyword (default)

        Args:
            query: The search query text
            limit: Maximum number of results to return
            alpha: Balance between vector (1) and keyword (0) search.
                  Defaults to settings.hybrid_search_alpha
            section_filter: Only search in this section type
                          (e.g., "adverse_reactions", "contraindications")
            drug_filter: Only search for this drug name

        Returns:
            List of result dictionaries with content and metadata
        """
        alpha = alpha if alpha is not None else settings.hybrid_search_alpha

        logger.debug(
            "hybrid_search_start",
            query=query[:50],
            limit=limit,
            alpha=alpha,
            section_filter=section_filter,
            drug_filter=drug_filter,
        )

        # Generate query embedding
        query_vector = self.embedder.embed_query(query)

        # Build filters
        filters = self._build_filters(section_filter, drug_filter)

        # Execute hybrid search
        results = self.collection.query.hybrid(
            query=query,
            vector=query_vector.tolist(),
            alpha=alpha,
            limit=limit,
            filters=filters,
            return_metadata=MetadataQuery(score=True),
        )

        # Format results
        formatted_results = []
        for obj in results.objects:
            formatted_results.append({
                "chunk_id": obj.properties.get("chunk_id"),
                "drug_name": obj.properties.get("drug_name"),
                "generic_name": obj.properties.get("generic_name"),
                "section_type": obj.properties.get("section_type"),
                "section_title": obj.properties.get("section_title"),
                "content": obj.properties.get("content"),
                "token_count": obj.properties.get("token_count"),
                "set_id": obj.properties.get("set_id"),
                "manufacturer": obj.properties.get("manufacturer"),
                "source_url": obj.properties.get("source_url"),
                "score": obj.metadata.score if obj.metadata else None,
            })

        logger.debug("hybrid_search_complete", results=len(formatted_results))

        return formatted_results

    def vector_search(
            self,
            query: str,
            limit: int = 10,
            section_filter: str | None = None,
            drug_filter: str | None = None,
    ) -> list[dict]:
        """
        Perform pure vector (semantic) search.

        Use this when you want to find conceptually similar content
        regardless of exact wording.

        Args:
            query: The search query text
            limit: Maximum number of results
            section_filter: Only search in this section type
            drug_filter: Only search for this drug name

        Returns:
            List of result dictionaries
        """
        query_vector = self.embedder.embed_query(query)
        filters = self._build_filters(section_filter, drug_filter)

        results = self.collection.query.near_vector(
            near_vector=query_vector.tolist(),
            limit=limit,
            filters=filters,
            return_metadata=MetadataQuery(distance=True),
        )

        formatted_results = []
        for obj in results.objects:
            formatted_results.append({
                "chunk_id": obj.properties.get("chunk_id"),
                "drug_name": obj.properties.get("drug_name"),
                "generic_name": obj.properties.get("generic_name"),
                "section_type": obj.properties.get("section_type"),
                "section_title": obj.properties.get("section_title"),
                "content": obj.properties.get("content"),
                "set_id": obj.properties.get("set_id"),
                "source_url": obj.properties.get("source_url"),
                "distance": obj.metadata.distance if obj.metadata else None,
            })

        return formatted_results

    def keyword_search(
            self,
            query: str,
            limit: int = 10,
            section_filter: str | None = None,
            drug_filter: str | None = None,
    ) -> list[dict]:
        """
        Perform pure keyword (BM25) search.

        Use this when you need exact term matching,
        like searching for specific drug names or codes.

        Args:
            query: The search query text
            limit: Maximum number of results
            section_filter: Only search in this section type
            drug_filter: Only search for this drug name

        Returns:
            List of result dictionaries
        """
        filters = self._build_filters(section_filter, drug_filter)

        results = self.collection.query.bm25(
            query=query,
            limit=limit,
            filters=filters,
            return_metadata=MetadataQuery(score=True),
        )

        formatted_results = []
        for obj in results.objects:
            formatted_results.append({
                "chunk_id": obj.properties.get("chunk_id"),
                "drug_name": obj.properties.get("drug_name"),
                "generic_name": obj.properties.get("generic_name"),
                "section_type": obj.properties.get("section_type"),
                "section_title": obj.properties.get("section_title"),
                "content": obj.properties.get("content"),
                "set_id": obj.properties.get("set_id"),
                "source_url": obj.properties.get("source_url"),
                "score": obj.metadata.score if obj.metadata else None,
            })

        return formatted_results

    def _build_filters(
            self,
            section_filter: str | None,
            drug_filter: str | None,
    ) -> Filter | None:
        """Build Weaviate filter from parameters."""
        filters = []

        if section_filter:
            filters.append(
                Filter.by_property("section_type").equal(section_filter)
            )

        if drug_filter:
            # Use contains_any for partial matching
            filters.append(
                Filter.by_property("drug_name").like(f"*{drug_filter}*")
            )

        if not filters:
            return None

        # Combine with AND
        combined = filters[0]
        for f in filters[1:]:
            combined = combined & f

        return combined

    def delete_all(self) -> None:
        """
        Delete all documents from the collection.

        Use for testing or resetting the database.
        """
        from drugquery.vectorstore.schema import delete_schema

        delete_schema(self.client)
        create_schema(self.client)
        self.collection = self.client.collections.get(COLLECTION_NAME)

        logger.info("collection_reset", name=COLLECTION_NAME)


if __name__ == "__main__":
    # Quick test: index some chunks and search
    from pathlib import Path
    from drugquery.logging import configure_logging
    from drugquery.ingestion.chunk import load_chunks_from_jsonl

    configure_logging()

    chunks_path = Path("data/processed/chunks.jsonl")

    if not chunks_path.exists():
        print(f"No chunks found at {chunks_path}")
        print("Run the ingestion pipeline first:")
        print("  uv run drugquery download --limit 50")
        print("  uv run drugquery ingest")
    else:
        # Load chunks
        chunks = load_chunks_from_jsonl(chunks_path)
        print(f"Loaded {len(chunks)} chunks")

        # Create store and index
        with DrugVectorStore() as store:
            # Check current state
            stats = store.get_stats()
            print(f"Current stats: {stats}")

            # Index if empty
            if stats["count"] == 0:
                print("\nIndexing chunks...")
                indexed = store.index_chunks(chunks)
                print(f"Indexed {indexed} chunks")

            # Test search
            print("\n" + "=" * 60)
            query = "What are common side effects?"
            print(f"Query: {query}")
            print("=" * 60)

            results = store.hybrid_search(query, limit=3)

            for i, result in enumerate(results, 1):
                print(f"\n--- Result {i} (score: {result['score']:.4f}) ---")
                print(f"Drug: {result['drug_name']}")
                print(f"Section: {result['section_type']}")
                preview = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
                print(f"Content: {preview}")