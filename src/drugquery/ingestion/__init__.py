"""
Data ingestion pipeline for FDA DailyMed SPL documents.

This module handles:
- Downloading SPL files from DailyMed
- Parsing SPL XML into structured data
- Chunking documents for optimal retrieval
"""

from drugquery.ingestion.download import download_drug_labels
from drugquery.ingestion.parse import parse_spl_file, DrugLabel, DrugSection
from drugquery.ingestion.chunk import DrugLabelChunker, DocumentChunk

__all__ = [
    "download_drug_labels",
    "parse_spl_file",
    "DrugLabel",
    "DrugSection",
    "DrugLabelChunker",
    "DocumentChunk",
]
