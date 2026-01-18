"""Pytest configuration and shared fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def sample_spl_path() -> Path:
    """Path to a sample SPL file for testing."""
    return Path(__file__).parent / "fixtures" / "sample_spl.xml"


@pytest.fixture
def sample_drug_label():
    """Sample parsed drug label for testing."""
    from drugquery.ingestion.parse import DrugLabel, DrugSection
    
    return DrugLabel(
        set_id="test-set-id-123",
        drug_name="Test Drug",
        generic_name="testdrugium",
        manufacturer="Test Pharma Inc.",
        ndc_codes=["12345-678-90"],
        sections=[
            DrugSection(
                section_type="indications_usage",
                section_code="34067-9",
                title="INDICATIONS AND USAGE",
                text="Test Drug is indicated for the treatment of test conditions.",
            ),
            DrugSection(
                section_type="contraindications",
                section_code="34070-3",
                title="CONTRAINDICATIONS",
                text="Test Drug is contraindicated in patients with known hypersensitivity.",
            ),
        ],
        raw_xml_path="/path/to/test.xml",
    )
