"""
Structured logging configuration using structlog.

Provides consistent, JSON-formatted logs for production
and human-readable logs for development.
"""

import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor

from drugquery.config import settings


def configure_logging() -> None:
    """Configure structlog for the application."""
    
    # Determine if we're in development
    is_dev = settings.environment == "development"
    
    # Shared processors for all environments
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    
    if is_dev:
        # Development: colorful, human-readable output
        processors: list[Processor] = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        # Production: JSON output for log aggregation
        processors = [
            *shared_processors,
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Also configure standard library logging for third-party libs
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.getLevelName(settings.log_level),
    )
    
    # Reduce noise from httpx and other verbose libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def get_logger(name: str | None = None, **initial_context: Any) -> structlog.BoundLogger:
    """
    Get a logger instance with optional initial context.
    
    Args:
        name: Optional logger name (typically __name__)
        **initial_context: Key-value pairs to bind to all log messages
        
    Returns:
        Configured structlog logger
        
    Example:
        logger = get_logger(__name__, component="ingestion")
        logger.info("processing_file", filename="drug.xml")
    """
    logger = structlog.get_logger(name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger
