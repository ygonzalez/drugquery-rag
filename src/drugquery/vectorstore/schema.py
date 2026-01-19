"""
Weaviate schema for drug label chunks.

The schema defines:
- Collection name and description
- Properties (fields) and their types
- Which properties are searchable (tokenized)
- Vector configuration

Key design decisions:
1. We provide our own vectors (no built-in vectorizer)
2. drug_name and generic_name are tokenized for keyword search
3. section_type uses "field" tokenization for exact filtering
4. content is the main text we embed and search

Usage:
    from drugquery.vectorstore.schema import create_schema, delete_schema

    create_schema(client)  # Create the collection
    delete_schema(client)  # Delete (for reset)
"""

import weaviate
from weaviate.classes.config import Configure, Property, DataType, Tokenization

from drugquery.logging import get_logger

logger = get_logger(__name__, component="schema")

# Collection name in Weaviate
COLLECTION_NAME = "DrugChunk"


def get_schema_properties() -> list[Property]:
    """
    Define the properties (fields) for drug chunks.

    Property types:
    - TEXT: String, can be tokenized for search
    - INT: Integer
    - TEXT_ARRAY: List of strings

    Tokenization options:
    - WORD: Split on whitespace/punctuation (for keyword search)
    - FIELD: No splitting (for exact match/filtering)
    - WHITESPACE: Split on whitespace only
    """
    return [
        # Unique identifier
        Property(
            name="chunk_id",
            data_type=DataType.TEXT,
            description="Unique identifier: {set_id}_{section_code}_{index}",
            tokenization=Tokenization.FIELD,  # Exact match only
        ),

        # Drug identification - tokenized for keyword search
        Property(
            name="drug_name",
            data_type=DataType.TEXT,
            description="Brand name of the drug",
            tokenization=Tokenization.WORD,  # Enable keyword search
        ),
        Property(
            name="generic_name",
            data_type=DataType.TEXT,
            description="Generic name of the drug",
            tokenization=Tokenization.WORD,
        ),

        # Section metadata
        Property(
            name="section_type",
            data_type=DataType.TEXT,
            description="Type: indications_usage, warnings, contraindications, etc.",
            tokenization=Tokenization.FIELD,  # For filtering
        ),
        Property(
            name="section_title",
            data_type=DataType.TEXT,
            description="Human-readable section title",
            tokenization=Tokenization.WORD,
        ),

        # Main content - this is what we embed and search
        Property(
            name="content",
            data_type=DataType.TEXT,
            description="The actual text content of the chunk",
            tokenization=Tokenization.WORD,
        ),

        # Chunk metadata
        Property(
            name="token_count",
            data_type=DataType.INT,
            description="Number of tokens in content",
        ),
        Property(
            name="chunk_index",
            data_type=DataType.INT,
            description="Position within the section (0-indexed)",
        ),
        Property(
            name="total_chunks",
            data_type=DataType.INT,
            description="Total chunks for this section",
        ),

        # Source attribution
        Property(
            name="set_id",
            data_type=DataType.TEXT,
            description="FDA SPL Set ID for source linking",
            tokenization=Tokenization.FIELD,
        ),
        Property(
            name="manufacturer",
            data_type=DataType.TEXT,
            description="Drug manufacturer",
            tokenization=Tokenization.WORD,
        ),
        Property(
            name="source_url",
            data_type=DataType.TEXT,
            description="URL to FDA DailyMed page",
            tokenization=Tokenization.FIELD,
        ),
    ]


def create_schema(client: weaviate.WeaviateClient, delete_existing: bool = False) -> None:
    """
    Create the DrugChunk collection in Weaviate.

    Args:
        client: Connected Weaviate client
        delete_existing: If True, delete existing collection first

    Raises:
        Exception: If collection already exists and delete_existing=False
    """
    # Check if collection exists
    if client.collections.exists(COLLECTION_NAME):
        if delete_existing:
            logger.warning("deleting_existing_collection", name=COLLECTION_NAME)
            client.collections.delete(COLLECTION_NAME)
        else:
            logger.info("collection_exists", name=COLLECTION_NAME)
            return

    # Create collection
    # We use "none" vectorizer because we provide our own embeddings
    client.collections.create(
        name=COLLECTION_NAME,
        description="Chunks from FDA drug labels for RAG retrieval",
        properties=get_schema_properties(),
        # No built-in vectorizer - we'll provide vectors ourselves
        vectorizer_config=Configure.Vectorizer.none(),
    )

    logger.info("collection_created", name=COLLECTION_NAME)


def delete_schema(client: weaviate.WeaviateClient) -> None:
    """
    Delete the DrugChunk collection.

    Use this to reset the database during development.
    """
    if client.collections.exists(COLLECTION_NAME):
        client.collections.delete(COLLECTION_NAME)
        logger.info("collection_deleted", name=COLLECTION_NAME)
    else:
        logger.info("collection_not_found", name=COLLECTION_NAME)


def get_collection_stats(client: weaviate.WeaviateClient) -> dict:
    """
    Get statistics about the collection.

    Returns:
        Dict with count and other stats
    """
    if not client.collections.exists(COLLECTION_NAME):
        return {"exists": False, "count": 0}

    collection = client.collections.get(COLLECTION_NAME)

    # Get count using aggregation
    result = collection.aggregate.over_all(total_count=True)

    return {
        "exists": True,
        "count": result.total_count,
        "name": COLLECTION_NAME,
    }


if __name__ == "__main__":
    # Quick test: create schema
    from drugquery.logging import configure_logging
    from drugquery.vectorstore.client import get_weaviate_client

    configure_logging()

    with get_weaviate_client() as client:
        # Show current state
        stats = get_collection_stats(client)
        print(f"Before: {stats}")

        # Create schema (delete existing for clean slate)
        create_schema(client, delete_existing=True)

        # Show new state
        stats = get_collection_stats(client)
        print(f"After: {stats}")