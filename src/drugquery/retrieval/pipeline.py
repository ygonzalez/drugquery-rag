"""
Complete retrieval pipeline combining all components.

Pipeline stages:
1. Query Expansion: Add medical synonyms (optional)
2. Hybrid Search: Fast initial retrieval
3. Reranking: Precise scoring of top results (optional)

This module orchestrates the full retrieval process and provides
a clean interface for the rest of the application.

Usage:
    from drugquery.retrieval import RetrievalPipeline

    pipeline = RetrievalPipeline()

    result = pipeline.retrieve("What are the side effects of aspirin?")

    for chunk in result.chunks:
        print(f"{chunk['drug_name']}: {chunk['content'][:100]}")
"""

from dataclasses import dataclass, field

from drugquery.vectorstore.client import DrugVectorStore
from drugquery.retrieval.query import QueryExpander
from drugquery.retrieval.rerank import Reranker, get_reranker
from drugquery.logging import get_logger

logger = get_logger(__name__, component="retrieval_pipeline")


@dataclass
class RetrievalResult:
    """
    Result from the retrieval pipeline.

    Contains the retrieved chunks plus metadata about the retrieval process
    for debugging and observability.
    """
    # The original user query
    query: str

    # Query after expansion (includes original + added terms)
    expanded_query: str

    # The retrieved chunks (list of dicts with content, metadata, scores)
    chunks: list[dict]

    # Metadata about the retrieval process
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        """Number of chunks retrieved."""
        return len(self.chunks)

    def __bool__(self) -> bool:
        """True if any chunks were retrieved."""
        return len(self.chunks) > 0


class RetrievalPipeline:
    """
    Complete retrieval pipeline for drug information.

    Combines:
    - Query expansion (optional): Adds medical terminology
    - Hybrid search: Vector + keyword search
    - Reranking (optional): Cross-encoder scoring

    Configuration:
        use_query_expansion: Whether to expand queries with Claude
        use_reranking: Whether to rerank with cross-encoder
        initial_k: How many results to get from initial search
        final_k: How many results to return after reranking

    Example:
        # Full pipeline (expansion + reranking)
        pipeline = RetrievalPipeline()

        # Faster pipeline (no expansion or reranking)
        pipeline = RetrievalPipeline(
            use_query_expansion=False,
            use_reranking=False,
        )

        result = pipeline.retrieve("side effects of ibuprofen")
        print(f"Found {len(result)} relevant chunks")
    """

    def __init__(
            self,
            use_query_expansion: bool = True,
            use_reranking: bool = True,
            initial_k: int = 20,
            final_k: int = 5,
    ):
        """
        Initialize the retrieval pipeline.

        Args:
            use_query_expansion: Whether to expand queries with medical terms
            use_reranking: Whether to rerank results with cross-encoder
            initial_k: Number of results from initial hybrid search
            final_k: Number of results to return after reranking
        """
        self.use_query_expansion = use_query_expansion
        self.use_reranking = use_reranking
        self.initial_k = initial_k
        self.final_k = final_k

        # Initialize components
        self.vectorstore = DrugVectorStore()

        # Only initialize what we need
        self.expander = QueryExpander() if use_query_expansion else None
        self.reranker = get_reranker() if use_reranking else None

        logger.info(
            "retrieval_pipeline_initialized",
            use_expansion=use_query_expansion,
            use_reranking=use_reranking,
            initial_k=initial_k,
            final_k=final_k,
        )

    def retrieve(
            self,
            query: str,
            section_filter: str | None = None,
            drug_filter: str | None = None,
    ) -> RetrievalResult:
        """
        Execute the full retrieval pipeline.

        Pipeline:
        1. Query expansion (if enabled)
        2. Hybrid search in Weaviate
        3. Reranking (if enabled)

        Args:
            query: The user's search query
            section_filter: Only search specific section type
                          (e.g., "adverse_reactions", "contraindications")
            drug_filter: Only search specific drug name

        Returns:
            RetrievalResult with chunks and metadata
        """
        logger.info("retrieval_start", query=query[:50])

        # Track what we did for observability
        metadata = {
            "original_query": query,
            "used_expansion": False,
            "used_reranking": False,
            "initial_results": 0,
            "final_results": 0,
        }

        # ===== Step 1: Query Expansion =====
        if self.expander:
            expanded_query = self.expander.expand_with_original(query)
            metadata["used_expansion"] = True
            metadata["expanded_terms"] = expanded_query.replace(query, "").strip()
            logger.debug("query_expanded", expanded=expanded_query[:100])
        else:
            expanded_query = query

        # ===== Step 2: Hybrid Search =====
        initial_results = self.vectorstore.hybrid_search(
            query=expanded_query,
            limit=self.initial_k,
            section_filter=section_filter,
            drug_filter=drug_filter,
        )

        metadata["initial_results"] = len(initial_results)
        logger.debug("hybrid_search_complete", count=len(initial_results))

        # If no results, return early
        if not initial_results:
            logger.info("retrieval_no_results", query=query[:50])
            return RetrievalResult(
                query=query,
                expanded_query=expanded_query,
                chunks=[],
                metadata=metadata,
            )

        # ===== Step 3: Reranking =====
        if self.reranker:
            final_results = self.reranker.rerank(
                query=query,  # Use original query for reranking (more precise)
                chunks=initial_results,
                top_k=self.final_k,
            )
            metadata["used_reranking"] = True
            logger.debug("reranking_complete", count=len(final_results))
        else:
            # No reranking - just take top final_k
            final_results = initial_results[:self.final_k]

        metadata["final_results"] = len(final_results)

        logger.info(
            "retrieval_complete",
            query=query[:50],
            initial=metadata["initial_results"],
            final=metadata["final_results"],
        )

        return RetrievalResult(
            query=query,
            expanded_query=expanded_query,
            chunks=final_results,
            metadata=metadata,
        )

    def close(self) -> None:
        """Close connections and clean up resources."""
        self.vectorstore.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience function for simple use cases
def retrieve(
        query: str,
        section_filter: str | None = None,
        drug_filter: str | None = None,
        use_expansion: bool = True,
        use_reranking: bool = True,
) -> RetrievalResult:
    """
    Simple function interface for retrieval.

    Creates a pipeline, retrieves, and cleans up.
    Use RetrievalPipeline directly for multiple queries.

    Args:
        query: Search query
        section_filter: Filter by section type
        drug_filter: Filter by drug name
        use_expansion: Whether to expand query
        use_reranking: Whether to rerank results

    Returns:
        RetrievalResult with chunks and metadata
    """
    with RetrievalPipeline(
            use_query_expansion=use_expansion,
            use_reranking=use_reranking,
    ) as pipeline:
        return pipeline.retrieve(
            query=query,
            section_filter=section_filter,
            drug_filter=drug_filter,
        )


# Quick test when run directly
if __name__ == "__main__":
    from drugquery.logging import configure_logging

    configure_logging()

    print("\nTesting Retrieval Pipeline")
    print("=" * 60)

    # Test with full pipeline
    with RetrievalPipeline() as pipeline:
        query = "What are the side effects?"

        print(f"\nQuery: {query}")

        result = pipeline.retrieve(query)

        print(f"\nExpanded query: {result.expanded_query}")
        print(f"Initial results: {result.metadata['initial_results']}")
        print(f"Final results: {result.metadata['final_results']}")
        print(f"Used expansion: {result.metadata['used_expansion']}")
        print(f"Used reranking: {result.metadata['used_reranking']}")

        print("\nTop results:")
        for i, chunk in enumerate(result.chunks, 1):
            score = chunk.get('rerank_score', chunk.get('score', 0))
            print(f"\n{i}. (score: {score:.4f})")
            print(f"   Drug: {chunk['drug_name']}")
            print(f"   Section: {chunk['section_type']}")
            print(f"   Content: {chunk['content'][:100]}...")