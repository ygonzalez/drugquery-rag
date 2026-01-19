"""
Text embeddings for semantic search.

Supports two embedding backends:
1. OpenAI (recommended): Higher quality, requires API key
2. Local (sentence-transformers): Free, runs locally

Usage:
    from drugquery.vectorstore.embeddings import DrugEmbedder

    # Use OpenAI (default if OPENAI_API_KEY is set)
    embedder = DrugEmbedder(provider="openai")

    # Use local model
    embedder = DrugEmbedder(provider="local")

    # Embed documents (for indexing)
    vectors = embedder.embed_documents(["text1", "text2"])

    # Embed query (for searching)
    query_vector = embedder.embed_query("What are the side effects?")
"""

from abc import ABC, abstractmethod

import numpy as np

from drugquery.config import settings
from drugquery.logging import get_logger

logger = get_logger(__name__, component="embeddings")


class BaseEmbedder(ABC):
    """Abstract base class for embedders."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed multiple documents."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        pass


class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI embeddings using text-embedding-3-small.

    Pros:
    - High quality embeddings
    - No local model download
    - Fast API response

    Cons:
    - Costs money (~$0.02 per 1M tokens)
    - Requires internet connection
    - API rate limits
    """

    # Model dimensions
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model_name: str = "text-embedding-3-small"):
        """
        Initialize OpenAI embedder.

        Args:
            model_name: OpenAI embedding model to use
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: uv add openai"
            )

        api_key = settings.openai_api_key
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Add it to your .env file."
            )

        self.client = OpenAI(api_key=api_key.get_secret_value())
        self.model_name = model_name
        self._dimension = self.MODEL_DIMENSIONS.get(model_name, 1536)

        logger.info("openai_embedder_initialized", model=model_name, dimension=self._dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(
            self,
            texts: list[str],
            batch_size: int = 100,
            show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed multiple documents using OpenAI API.

        Args:
            texts: List of texts to embed
            batch_size: Texts per API call (max 2048)
            show_progress: Whether to log progress

        Returns:
            numpy array of shape (len(texts), dimension)
        """
        if not texts:
            return np.array([])

        logger.info("embedding_documents_openai", count=len(texts))

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
            )

            # Extract embeddings in order
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

            if show_progress and (i + batch_size) % 500 == 0:
                logger.info("embedding_progress", completed=min(i + batch_size, len(texts)), total=len(texts))

        embeddings = np.array(all_embeddings)
        logger.info("embedding_complete_openai", shape=embeddings.shape)

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=query,
        )
        return np.array(response.data[0].embedding)


class LocalEmbedder(BaseEmbedder):
    """
    Local embeddings using sentence-transformers.

    Pros:
    - Free (no API costs)
    - Works offline
    - No rate limits

    Cons:
    - Requires model download (~400MB)
    - Slower on CPU
    - Slightly lower quality than OpenAI
    """

    def __init__(self, model_name: str | None = None):
        """
        Initialize local embedder.

        Args:
            model_name: HuggingFace model name. Defaults to settings.embedding_model
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Install with: uv add sentence-transformers"
            )

        self.model_name = model_name or settings.embedding_model

        logger.info("loading_local_embedding_model", model=self.model_name)

        self.model = SentenceTransformer(self.model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()

        logger.info("local_embedder_initialized", model=self.model_name, dimension=self._dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(
            self,
            texts: list[str],
            batch_size: int = 32,
            show_progress: bool = True,
    ) -> np.ndarray:
        """Embed multiple documents."""
        if not texts:
            return np.array([])

        # BGE models benefit from instruction prefix
        if "bge" in self.model_name.lower():
            texts = [f"Represent this drug information for retrieval: {t}" for t in texts]

        logger.info("embedding_documents_local", count=len(texts))

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )

        logger.info("embedding_complete_local", shape=embeddings.shape)

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        if "bge" in self.model_name.lower():
            query = f"Represent this question for retrieving relevant drug information: {query}"

        return self.model.encode(query, normalize_embeddings=True)


class DrugEmbedder:
    """
    Unified embedder that supports multiple backends.

    Automatically selects OpenAI if OPENAI_API_KEY is set,
    otherwise falls back to local embeddings.

    Example:
        # Auto-select based on available API key
        embedder = DrugEmbedder()

        # Force OpenAI
        embedder = DrugEmbedder(provider="openai")

        # Force local
        embedder = DrugEmbedder(provider="local")
    """

    def __init__(
            self,
            provider: str | None = None,
            model_name: str | None = None,
    ):
        """
        Initialize the embedder.

        Args:
            provider: "openai" or "local". If None, auto-selects based on API key.
            model_name: Model name (provider-specific)
        """
        # Auto-select provider if not specified
        if provider is None:
            if settings.openai_api_key:
                provider = "openai"
                logger.info("auto_selected_provider", provider="openai")
            else:
                provider = "local"
                logger.info("auto_selected_provider", provider="local")

        self.provider = provider

        # Initialize the appropriate embedder
        if provider == "openai":
            self._embedder = OpenAIEmbedder(
                model_name=model_name or "text-embedding-3-small"
            )
        elif provider == "local":
            self._embedder = LocalEmbedder(model_name=model_name)
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'local'.")

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._embedder.dimension

    @property
    def model_name(self) -> str:
        """Return model name."""
        if isinstance(self._embedder, OpenAIEmbedder):
            return self._embedder.model_name
        return self._embedder.model_name

    def embed_documents(
            self,
            texts: list[str],
            batch_size: int = 32,
            show_progress: bool = True,
    ) -> np.ndarray:
        """Embed multiple documents."""
        return self._embedder.embed_documents(texts, batch_size, show_progress)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        return self._embedder.embed_query(query)


# Singleton instance for reuse
_embedder: DrugEmbedder | None = None


def get_embedder(provider: str | None = None) -> DrugEmbedder:
    """
    Get a shared embedder instance.

    Args:
        provider: Force a specific provider ("openai" or "local")

    Returns:
        Shared DrugEmbedder instance
    """
    global _embedder
    if _embedder is None or (provider and _embedder.provider != provider):
        _embedder = DrugEmbedder(provider=provider)
    return _embedder


if __name__ == "__main__":
    # Quick test
    from drugquery.logging import configure_logging

    configure_logging()

    # Test with auto-selected provider
    embedder = DrugEmbedder()

    print(f"\nProvider: {embedder.provider}")
    print(f"Model: {embedder.model_name}")
    print(f"Dimension: {embedder.dimension}")

    # Test embedding
    docs = [
        "Aspirin is used to treat pain and fever.",
        "Do not take aspirin if allergic to NSAIDs.",
    ]
    query = "What is aspirin used for?"

    doc_embeddings = embedder.embed_documents(docs)
    query_embedding = embedder.embed_query(query)

    print(f"\nDocument embeddings shape: {doc_embeddings.shape}")
    print(f"Query embedding shape: {query_embedding.shape}")

    # Calculate similarities
    similarities = np.dot(doc_embeddings, query_embedding)
    print(f"\nSimilarities to query '{query}':")
    for doc, sim in zip(docs, similarities):
        print(f"  {sim:.4f}: {doc}")