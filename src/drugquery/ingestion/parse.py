"""
Parse FDA SPL (Structured Product Labeling) XML documents.

SPL is an XML format defined by HL7 for drug labels. Each document contains:
- Drug identification (names, NDC codes, manufacturer)
- Structured sections identified by LOINC codes
- Clinical information (indications, warnings, dosage, etc.)

Documentation:
- https://www.fda.gov/industry/fda-resources-data-standards/structured-product-labeling-resources
- https://loinc.org/ (for section codes)

Usage:
    from drugquery.ingestion.parse import parse_spl_file

    drug_label = parse_spl_file(Path("data/raw/some-setid.xml"))
    print(drug_label.drug_name)
    for section in drug_label.sections:
        print(f"{section.section_type}: {section.title}")
"""

from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

from drugquery.logging import get_logger

logger = get_logger(__name__, component="parse")

# ============================================================================
# LOINC Section Codes
# ============================================================================
# These codes identify what type of content each section contains.
# The FDA uses standardized LOINC codes so we can reliably extract sections.
#
# Full list: https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-sections.cfm
# ============================================================================

SECTION_CODES = {
    # High-priority sections for drug information
    "34066-1": "boxed_warning",           # Black box warnings (most severe)
    "34067-9": "indications_usage",       # What the drug treats
    "34068-7": "dosage_administration",   # How to take it
    "34070-3": "contraindications",       # When NOT to use
    "34071-1": "warnings_precautions",    # Important warnings
    "43685-7": "warnings_and_cautions",   # Alternative warning section
    "34084-4": "adverse_reactions",       # Side effects
    "34073-7": "drug_interactions",       # Interactions with other drugs

    # Additional useful sections
    "34076-0": "clinical_pharmacology",   # How the drug works
    "34090-1": "clinical_studies",        # Study results
    "34069-5": "how_supplied",            # Packaging/forms available
    "42232-9": "precautions",             # General precautions
    "43684-0": "use_specific_populations", # Pregnancy, nursing, etc.
    "34074-5": "drug_lab_interactions",   # Lab test interactions
    "34088-5": "overdosage",              # Overdose information
    "51945-4": "package_label",           # Label information
    "42229-5": "spl_patient_package_insert", # Patient information
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DrugSection:
    """
    A section from a drug label (e.g., indications, warnings).

    Each section has:
    - section_type: Normalized name (e.g., "indications_usage")
    - section_code: LOINC code (e.g., "34067-9")
    - title: Human-readable title from the document
    - text: The actual content
    """
    section_type: str
    section_code: str
    title: str
    text: str


@dataclass
class DrugLabel:
    """
    Parsed drug label containing all sections and metadata.

    This is the main data structure we pass to the chunking step.
    """
    set_id: str
    drug_name: str
    generic_name: str | None
    manufacturer: str | None
    ndc_codes: list[str] = field(default_factory=list)
    sections: list[DrugSection] = field(default_factory=list)
    raw_xml_path: str = ""


# ============================================================================
# XML Namespace Handling
# ============================================================================
# SPL documents use the HL7 namespace. We need to include it in all XPath queries.

SPL_NAMESPACE = {"spl": "urn:hl7-org:v3"}


def _xpath(element: etree._Element, path: str) -> list:
    """Helper to run XPath with SPL namespace."""
    return element.xpath(path, namespaces=SPL_NAMESPACE)


def _xpath_first(element: etree._Element, path: str) -> etree._Element | None:
    """Helper to get first XPath result or None."""
    results = _xpath(element, path)
    return results[0] if results else None


def _xpath_text(element: etree._Element, path: str) -> str | None:
    """Helper to get text content from XPath result."""
    elem = _xpath_first(element, path)
    if elem is not None:
        return elem.text
    return None


# ============================================================================
# Extraction Functions
# ============================================================================

def _extract_set_id(root: etree._Element) -> str:
    """Extract the unique SetID from the document."""
    set_id_elem = _xpath_first(root, ".//spl:setId")
    if set_id_elem is not None:
        return set_id_elem.get("root", "unknown")
    return "unknown"


def _extract_drug_name(root: etree._Element) -> str:
    """
    Extract the drug name from the document.

    SPL documents can store the name in several places:
    1. manufacturedProduct/name
    2. Document title
    3. Subject/name

    We try each in order of preference.
    """
    # Try manufactured product name first (most reliable)
    name = _xpath_text(root, ".//spl:manufacturedProduct/spl:name")
    if name:
        return name.strip()

    # Try subject name
    name = _xpath_text(root, ".//spl:subject/spl:manufacturedProduct/spl:name")
    if name:
        return name.strip()

    # Fall back to document title
    name = _xpath_text(root, ".//spl:title")
    if name:
        # Title often has extra info, try to extract just the drug name
        # e.g., "ASPIRIN- aspirin tablet" -> "ASPIRIN"
        return name.split("-")[0].strip()

    return "Unknown Drug"


def _extract_generic_name(root: etree._Element) -> str | None:
    """Extract the generic (non-brand) name if available."""
    # Generic name is often in asEntityWithGeneric
    generic = _xpath_text(root, ".//spl:asEntityWithGeneric/spl:genericMedicine/spl:name")
    if generic:
        return generic.strip()

    # Sometimes it's in the activeIngredient
    generic = _xpath_text(root, ".//spl:activeIngredient/spl:activeIngredientSubstance/spl:name")
    if generic:
        return generic.strip()

    return None


def _extract_manufacturer(root: etree._Element) -> str | None:
    """Extract the manufacturer/labeler name."""
    # Try representedOrganization first
    mfr = _xpath_text(root, ".//spl:representedOrganization/spl:name")
    if mfr:
        return mfr.strip()

    # Try author/assignedEntity
    mfr = _xpath_text(root, ".//spl:author/spl:assignedEntity/spl:representedOrganization/spl:name")
    if mfr:
        return mfr.strip()

    return None


def _extract_ndc_codes(root: etree._Element) -> list[str]:
    """
    Extract NDC (National Drug Code) identifiers.

    NDC is a unique identifier for drugs in the US market.
    Format: labeler-product-package (e.g., 12345-678-90)
    """
    ndc_codes = []

    # NDC codes are in containerPackagedProduct or manufacturedProduct
    for code_elem in _xpath(root, ".//spl:code[@codeSystem='2.16.840.1.113883.6.69']"):
        code = code_elem.get("code")
        if code:
            ndc_codes.append(code)

    return list(set(ndc_codes))  # Remove duplicates


def _extract_section_text(section: etree._Element) -> str:
    """
    Extract all text content from a section.

    SPL sections can have nested structure:
    - <text> elements with paragraphs
    - <list> elements with items
    - Nested <content> elements

    We extract all text recursively and clean it up.
    """
    text_parts = []

    # Find all text elements in the section
    for text_elem in _xpath(section, ".//spl:text"):
        # Get all text content recursively
        content = etree.tostring(text_elem, method="text", encoding="unicode")
        if content:
            # Clean up whitespace
            cleaned = " ".join(content.split())
            if cleaned:
                text_parts.append(cleaned)

    # Join with paragraph breaks
    return "\n\n".join(text_parts)


def _extract_sections(root: etree._Element) -> list[DrugSection]:
    """
    Extract all sections from the document.

    We look for sections with known LOINC codes and extract their content.
    Sections without codes or with empty content are skipped.
    """
    sections = []

    # Find all section elements
    for section in _xpath(root, ".//spl:component/spl:section"):
        # Get the section code
        code_elem = _xpath_first(section, "spl:code")
        if code_elem is None:
            continue

        code = code_elem.get("code", "")

        # Look up the section type
        section_type = SECTION_CODES.get(code, "other")

        # Skip unknown sections to reduce noise
        if section_type == "other":
            continue

        # Get the title
        title_elem = _xpath_first(section, "spl:title")
        title = title_elem.text if title_elem is not None and title_elem.text else section_type.replace("_", " ").title()

        # Extract text content
        text = _extract_section_text(section)

        # Skip empty sections
        if not text.strip():
            continue

        sections.append(DrugSection(
            section_type=section_type,
            section_code=code,
            title=title.strip(),
            text=text.strip(),
        ))

    return sections


# ============================================================================
# Main Parsing Function
# ============================================================================

def parse_spl_file(xml_path: Path) -> DrugLabel:
    """
    Parse an SPL XML file into a structured DrugLabel.

    This is the main entry point for parsing. It:
    1. Reads and parses the XML file
    2. Extracts metadata (drug name, manufacturer, etc.)
    3. Extracts all known sections
    4. Returns a DrugLabel object ready for chunking

    Args:
        xml_path: Path to the SPL XML file

    Returns:
        Parsed DrugLabel with all sections and metadata

    Raises:
        FileNotFoundError: If the XML file doesn't exist
        etree.XMLSyntaxError: If the XML is malformed

    Example:
        >>> label = parse_spl_file(Path("data/raw/abc123.xml"))
        >>> print(f"Drug: {label.drug_name}")
        >>> print(f"Sections: {len(label.sections)}")
    """
    logger.debug("parsing_file", path=str(xml_path))

    # Parse XML
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    # Extract all components
    set_id = _extract_set_id(root)
    drug_name = _extract_drug_name(root)
    generic_name = _extract_generic_name(root)
    manufacturer = _extract_manufacturer(root)
    ndc_codes = _extract_ndc_codes(root)
    sections = _extract_sections(root)

    label = DrugLabel(
        set_id=set_id,
        drug_name=drug_name,
        generic_name=generic_name,
        manufacturer=manufacturer,
        ndc_codes=ndc_codes,
        sections=sections,
        raw_xml_path=str(xml_path),
    )

    logger.debug(
        "parsed_file",
        drug_name=drug_name,
        sections=len(sections),
        set_id=set_id,
    )

    return label


def parse_all_spl_files(input_dir: Path) -> list[DrugLabel]:
    """
    Parse all SPL XML files in a directory.

    Args:
        input_dir: Directory containing XML files

    Returns:
        List of parsed DrugLabel objects
    """
    labels = []
    xml_files = list(input_dir.glob("*.xml"))

    logger.info("parsing_directory", path=str(input_dir), file_count=len(xml_files))

    for i, xml_path in enumerate(xml_files):
        try:
            label = parse_spl_file(xml_path)
            labels.append(label)

            # Progress logging
            if (i + 1) % 100 == 0:
                logger.info("parse_progress", completed=i + 1, total=len(xml_files))

        except Exception as e:
            logger.warning("parse_failed", path=str(xml_path), error=str(e))
            continue

    logger.info("parsing_complete", parsed=len(labels), total=len(xml_files))

    return labels


if __name__ == "__main__":
    # Quick test: parse downloaded files
    from drugquery.logging import configure_logging

    configure_logging()

    input_dir = Path("data/raw")
    xml_files = list(input_dir.glob("*.xml"))

    if not xml_files:
        print("No XML files found. Run download first:")
        print("  uv run python -m drugquery.ingestion.download")
    else:
        # Parse first file as example
        label = parse_spl_file(xml_files[0])

        print(f"\n{'='*60}")
        print(f"Drug Name: {label.drug_name}")
        print(f"Generic Name: {label.generic_name}")
        print(f"Manufacturer: {label.manufacturer}")
        print(f"SetID: {label.set_id}")
        print(f"NDC Codes: {label.ndc_codes}")
        print(f"{'='*60}")
        print(f"\nSections ({len(label.sections)}):")
        for section in label.sections:
            preview = section.text[:100] + "..." if len(section.text) > 100 else section.text
            print(f"\n  [{section.section_code}] {section.title}")
            print(f"      {preview}")