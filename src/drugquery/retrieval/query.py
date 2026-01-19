"""
Query expansion for medical terminology.

Problem: Users use everyday language, but medical documents use clinical terms.
- User: "heart medicine side effects"
- Document: "cardiovascular drug adverse reactions"

Solution: Use an LLM to expand the query with synonyms and related terms.

Example:
    Input:  "heart medicine side effects"
    Output: "heart cardiac cardiovascular medicine drug medication
             side effects adverse reactions"

Usage:
    from drugquery.retrieval.query import QueryExpander

    expander = QueryExpander()
    expanded = expander.expand("What helps with high blood pressure?")
    # "high blood pressure hypertension antihypertensive medication treatment"
"""

from anthropic import Anthropic

from drugquery.config import settings
from drugquery.logging import get_logger

logger = get_logger(__name__, component="query_expansion")

# The prompt that tells Claude how to expand queries
EXPANSION_PROMPT = """You are a medical terminology assistant. Your job is to expand a user's question with relevant medical synonyms and related terms.

Given this question about medications:
{query}

Add relevant medical/clinical synonyms and related terms that might appear in FDA drug labels. Include:
1. Medical synonyms (e.g., "high blood pressure" → "hypertension")
2. Related drug class terms (e.g., "pain medicine" → "analgesic", "NSAID")
3. Clinical terminology (e.g., "side effects" → "adverse reactions")

Return ONLY the expanded query terms, no explanation. Keep it concise (under 50 words).
Do not repeat the original query - just add new relevant terms."""


class QueryExpander:
    """
    Expand user queries with medical terminology.

    Uses Claude Haiku for fast, cheap query expansion.

    Why Haiku?
    - This is a simple task (synonym generation)
    - We want low latency (user is waiting)
    - We want low cost (called on every query)

    Example:
        expander = QueryExpander()

        original = "What helps with headaches?"
        expanded = expander.expand(original)
        # Returns: "headache migraine cephalgia pain relief analgesic"

        # Use both for search
        search_query = f"{original} {expanded}"
    """

    def __init__(self, model: str | None = None):
        """
        Initialize the query expander.

        Args:
            model: Claude model to use. Defaults to settings.llm_model_fast (Haiku)
        """
        self.client = Anthropic(
            api_key=settings.anthropic_api_key.get_secret_value()
        )
        self.model = model or settings.llm_model_fast

        logger.info("query_expander_initialized", model=self.model)

    def expand(self, query: str) -> str:
        """
        Expand a query with medical terminology.

        Args:
            query: The user's original query

        Returns:
            Additional terms to add to the search (not including original query)
        """
        logger.debug("expanding_query", original=query[:50])

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=100,  # Keep it short
                messages=[{
                    "role": "user",
                    "content": EXPANSION_PROMPT.format(query=query)
                }]
            )

            expanded_terms = response.content[0].text.strip()

            logger.debug(
                "query_expanded",
                original=query[:50],
                expanded=expanded_terms[:50]
            )

            return expanded_terms

        except Exception as e:
            # If expansion fails, log and return empty string
            # The original query will still work
            logger.warning("query_expansion_failed", error=str(e))
            return ""

    def expand_with_original(self, query: str) -> str:
        """
        Expand query and combine with original.

        This is what you'd typically use for search:
        the original query plus expanded terms.

        Args:
            query: The user's original query

        Returns:
            Combined query: original + expanded terms
        """
        expanded = self.expand(query)
        if expanded:
            return f"{query} {expanded}"
        return query


# Quick test when run directly
if __name__ == "__main__":
    from drugquery.logging import configure_logging

    configure_logging()

    expander = QueryExpander()

    # Test queries
    test_queries = [
        "What helps with high blood pressure?",
        "Can I take this with alcohol?",
        "What are the side effects?",
        "Is it safe during pregnancy?",
        "How much should I take?",
    ]

    print("\nQuery Expansion Examples:")
    print("=" * 60)

    for query in test_queries:
        expanded = expander.expand(query)
        print(f"\nOriginal: {query}")
        print(f"Expanded: {expanded}")