"""Tests for configuration loading."""

import os
import pytest


def test_settings_loads_defaults():
    """Test that settings load with default values."""
    # Temporarily set required env var
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    
    # Clear cached settings
    from drugquery.config import get_settings
    get_settings.cache_clear()
    
    settings = get_settings()
    
    assert settings.weaviate_host == "localhost"
    assert settings.weaviate_port == 8080
    assert settings.chunk_max_tokens == 512
    assert settings.llm_model == "claude-sonnet-4-20250514"


def test_weaviate_url_property():
    """Test Weaviate URL construction."""
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    
    from drugquery.config import get_settings
    get_settings.cache_clear()
    
    settings = get_settings()
    
    assert settings.weaviate_url == "http://localhost:8080"
