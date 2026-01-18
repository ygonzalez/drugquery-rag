"""
Configuration management using pydantic-settings.

Loads configuration from environment variables and .env file.
"""

from functools import lru_cache
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -----------------
    # LLM API Keys
    # -----------------
    anthropic_api_key: SecretStr = Field(
        ...,
        description="Anthropic API key for Claude models",
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="Optional OpenAI API key for embeddings comparison",
    )

    # -----------------
    # LangSmith
    # -----------------
    langchain_api_key: SecretStr | None = Field(
        default=None,
        description="LangSmith API key for observability",
    )
    langchain_tracing_v2: bool = Field(
        default=True,
        description="Enable LangSmith tracing",
    )
    langchain_project: str = Field(
        default="drugquery-rag",
        description="LangSmith project name",
    )

    # -----------------
    # Weaviate
    # -----------------
    weaviate_host: str = Field(
        default="localhost",
        description="Weaviate host",
    )
    weaviate_port: int = Field(
        default=8080,
        description="Weaviate port",
    )
    weaviate_api_key: str | None = Field(
        default=None,
        description="Weaviate API key (empty for local anonymous access)",
    )

    # -----------------
    # Application
    # -----------------
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )
    environment: str = Field(
        default="development",
        description="Environment: development, staging, production",
    )

    # -----------------
    # Models
    # -----------------
    llm_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Main generation model",
    )
    llm_model_fast: str = Field(
        default="claude-3-haiku-20240307",
        description="Fast model for query expansion, guardrails",
    )
    embedding_model: str = Field(
        default="BAAI/bge-base-en-v1.5",
        description="Embedding model name",
    )
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Reranking model name",
    )

    # -----------------
    # RAG Configuration
    # -----------------
    chunk_max_tokens: int = Field(
        default=512,
        description="Maximum tokens per chunk",
    )
    chunk_overlap_tokens: int = Field(
        default=50,
        description="Overlap tokens between chunks",
    )
    retrieval_initial_k: int = Field(
        default=20,
        description="Initial retrieval count before reranking",
    )
    retrieval_final_k: int = Field(
        default=5,
        description="Final number of chunks after reranking",
    )
    hybrid_search_alpha: float = Field(
        default=0.7,
        description="Hybrid search alpha (0=keyword, 1=vector)",
    )

    # -----------------
    # API
    # -----------------
    api_host: str = Field(
        default="0.0.0.0",
        description="API host",
    )
    api_port: int = Field(
        default=8000,
        description="API port",
    )

    @property
    def weaviate_url(self) -> str:
        """Construct Weaviate URL."""
        return f"http://{self.weaviate_host}:{self.weaviate_port}"


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()


# Convenience access
settings = get_settings()
