"""
Cross-encoder reranking for improved precision.

Problem: Initial retrieval (bi-encoder) is fast but can miss nuance.
- Bi-encoder: Encode query and doc separately, compare vectors
- Cross-encoder: Encode query+doc together, output relevance score

The cross-encoder sees both texts at once, so it can understand:
- Whether the doc actually answers the question
- Subtle relevance signals
- Negation and context

Strategy:
1. Initial retrieval: Get top 20 results quickly (bi-encoder/hybrid)
2. Reranking: Score those 20 with cross-encoder (slower but better)
3. Return: Top 5 after reranking

Usage:
    from drugquery.retrieval.rerank import Reranker

    reranker = Reranker()
    reranked = reranker.rerank(
        query="What are the side effects?",
        chunks=initial_results,  # List of dicts with 'content'
        top_k=5,
    )
"""

from drugquery.config import settings
from drugquery.logging import get_logger

logger = get_logger(__name__, component="rerank")


class Reranker:
    """
    Rerank search results using a cross-encoder model.

    Cross-encoders are more accurate than bi-encoders because they
    see the query and document together. But they're slower, so we
    only use them on the top-K initial results.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    - Trained on MS MARCO passage ranking dataset
    - Good balance of speed and quality
    - ~22M parameters (small and fast)

    Example:
        reranker = Reranker()

        # Get initial results from hybrid search
        initial_results = store.hybrid_search(query, limit=20)

        # Rerank to get the best 5
        reranked = reranker.rerank(query, initial_results, top_k=5)
    """

    def __init__(self, model_name: str | None = None):
        """
        Initialize the reranker.

        Args:
            model_name: Cross-encoder model from HuggingFace.
                       Defaults to settings.rerank_model
        """
        # Import here to avoid loading if not used
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers required for reranking. "
                "Install with: uv add sentence-transformers"
            )

        self.model_name = model_name or settings.rerank_model

        logger.info("loading_rerank_model", model=self.model_name)

        # Load the cross-encoder model
        # This downloads the model on first use (~80MB)
        self.model = CrossEncoder(self.model_name)

        logger.info("reranker_initialized", model=self.model_name)

    def rerank(
            self,
            query: str,
            chunks: list[dict],
            top_k: int = 5,
    ) -> list[dict]:
        """
        Rerank chunks by relevance to the query.

        Process:
        1. Create (query, document) pairs
        2. Score each pair with cross-encoder
        3. Sort by score descending
        4. Return top_k results

        Args:
            query: The search query
            chunks: List of chunk dictionaries (must have 'content' key)
            top_k: Number of top results to return

        Returns:
            List of chunks sorted by relevance, with 'rerank_score' added
        """
        if not chunks:
            return []

        logger.debug("reranking_start", query=query[:50], num_chunks=len(chunks))

        # Step 1: Create query-document pairs
        # Cross-encoder needs to see both texts together
        pairs = []
        for chunk in chunks:
            content = chunk.get("content", "")
            pairs.append((query, content))

        # Step 2: Score with cross-encoder
        # Returns a score for each pair (higher = more relevant)
        scores = self.model.predict(pairs)

        # Step 3: Add scores to chunks and sort
        scored_chunks = []
        for chunk, score in zip(chunks, scores):
            chunk_with_score = chunk.copy()
            chunk_with_score["rerank_score"] = float(score)
            scored_chunks.append(chunk_with_score)

        # Sort by rerank_score (highest first)
        scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Step 4: Return top_k
        result = scored_chunks[:top_k]

        logger.debug(
            "reranking_complete",
            input_count=len(chunks),
            output_count=len(result),
            top_score=result[0]["rerank_score"] if result else None,
        )

        return result


# Singleton instance (model takes time to load)
_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    """
    Get a shared reranker instance.

    The model is loaded once and reused to avoid repeated loading.
    """
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


# Quick test when run directly
if __name__ == "__main__":
    from drugquery.logging import configure_logging

    configure_logging()

    # Create reranker
    reranker = Reranker()

    # Simulate search results
    query = "What are the side effects of aspirin?"

    fake_chunks = [
        {"content": "Aspirin may cause stomach upset, heartburn, and nausea.", "drug_name": "Aspirin"},
        {"content": "Store at room temperature away from moisture.", "drug_name": "Aspirin"},
        {"content": "Common adverse reactions include gastrointestinal bleeding.", "drug_name": "Aspirin"},
        {"content": "Take with food to reduce stomach irritation.", "drug_name": "Aspirin"},
        {"content": "The recommended dosage is 325mg to 650mg every 4 hours.", "drug_name": "Aspirin"},
    ]

    print(f"\nQuery: {query}")
    print("\nBefore reranking:")
    for i, chunk in enumerate(fake_chunks, 1):
        print(f"  {i}. {chunk['content'][:60]}...")

    # Rerank
    reranked = reranker.rerank(query, fake_chunks, top_k=3)

    print("\nAfter reranking (top 3):")
    for i, chunk in enumerate(reranked, 1):
        print(f"  {i}. (score: {chunk['rerank_score']:.4f}) {chunk['content'][:50]}...")