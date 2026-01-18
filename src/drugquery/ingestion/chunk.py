"""
Intelligent chunking for drug label documents.

Chunking is critical for RAG quality. The strategy:
1. Preserve section boundaries where possible (sections are natural units)
2. Split large sections at paragraph boundaries (not mid-sentence)
3. Include overlap for context continuity
4. Preserve rich metadata in every chunk (for filtering and attribution)

Why these choices matter:
- Section boundaries: A "warnings" chunk shouldn't bleed into "dosage"
- Paragraph splits: Keeps semantic units together
- Overlap: Helps when relevant info spans chunk boundaries
- Metadata: Enables filtering ("only search warnings") and citations

Usage:
    from drugquery.ingestion.chunk import DrugLabelChunker

    chunker = DrugLabelChunker(max_tokens=512)
    chunks = chunker.chunk_drug_label(drug_label)
"""

from dataclasses import dataclass, field

import tiktoken

from drugquery.config import settings
from drugquery.ingestion.parse import DrugLabel, DrugSection
from drugquery.logging import get_logger

logger = get_logger(__name__, component="chunk")


@dataclass
class DocumentChunk:
    """
    A chunk of a drug label optimized for retrieval.

    Each chunk contains:
    - Content: The actual text to embed and search
    - Metadata: Information for filtering and citation
    - Position info: Where this chunk fits in the original

    The chunk_id format is: {set_id}_{section_code}_{chunk_index}
    This ensures globally unique IDs across all documents.
    """
    chunk_id: str
    drug_name: str
    generic_name: str | None
    section_type: str
    section_title: str
    content: str
    token_count: int
    chunk_index: int      # 0-indexed position within section
    total_chunks: int     # Total chunks for this section
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"DocumentChunk(id={self.chunk_id}, tokens={self.token_count}, preview='{preview}')"


class DrugLabelChunker:
    """
    Chunk drug labels for optimal retrieval.

    The chunker balances two competing goals:
    1. Small chunks for precise retrieval
    2. Large enough chunks to preserve context

    Configuration:
    - max_tokens: Maximum tokens per chunk (default 512)
    - overlap_tokens: Tokens to overlap between chunks (default 50)
    - min_chunk_tokens: Don't create tiny chunks (default 100)

    Example:
        chunker = DrugLabelChunker(max_tokens=512)

        for label in drug_labels:
            chunks = chunker.chunk_drug_label(label)
            print(f"{label.drug_name}: {len(chunks)} chunks")
    """

    def __init__(
        self,
        max_tokens: int | None = None,
        overlap_tokens: int | None = None,
        min_chunk_tokens: int = 100,
    ):
        """
        Initialize the chunker.

        Args:
            max_tokens: Maximum tokens per chunk. Defaults to settings.chunk_max_tokens
            overlap_tokens: Overlap between chunks. Defaults to settings.chunk_overlap_tokens
            min_chunk_tokens: Minimum tokens to create a chunk
        """
        self.max_tokens = max_tokens or settings.chunk_max_tokens
        self.overlap_tokens = overlap_tokens or settings.chunk_overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens

        # Use cl100k_base tokenizer (same as GPT-4, Claude uses similar)
        # This gives us accurate token counts for our content
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        logger.info(
            "chunker_initialized",
            max_tokens=self.max_tokens,
            overlap_tokens=self.overlap_tokens,
        )

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.tokenizer.encode(text))

    def _create_chunk(
        self,
        label: DrugLabel,
        section: DrugSection,
        content: str,
        chunk_index: int,
        total_chunks: int,
    ) -> DocumentChunk:
        """Create a DocumentChunk with full metadata."""
        # Generate unique ID
        chunk_id = f"{label.set_id}_{section.section_code}_{chunk_index}"

        return DocumentChunk(
            chunk_id=chunk_id,
            drug_name=label.drug_name,
            generic_name=label.generic_name,
            section_type=section.section_type,
            section_title=section.title,
            content=content,
            token_count=self._count_tokens(content),
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            metadata={
                "set_id": label.set_id,
                "section_code": section.section_code,
                "manufacturer": label.manufacturer,
                "ndc_codes": label.ndc_codes,
                "source": "fda_dailymed",
                "source_url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={label.set_id}",
            },
        )

    def _chunk_section(
        self,
        label: DrugLabel,
        section: DrugSection,
    ) -> list[DocumentChunk]:
        """
        Chunk a single section.

        Algorithm:
        1. If section fits in one chunk, keep it whole
        2. Otherwise, split at paragraph boundaries
        3. Add overlap between chunks for context
        """
        text = section.text
        token_count = self._count_tokens(text)

        # Case 1: Small enough to keep whole
        if token_count <= self.max_tokens:
            return [self._create_chunk(label, section, text, 0, 1)]

        # Case 2: Need to split
        # Split into paragraphs (double newline is paragraph boundary)
        paragraphs = text.split("\n\n")

        # If no paragraph breaks, split on single newlines
        if len(paragraphs) == 1:
            paragraphs = text.split("\n")

        # If still just one block, split on sentences (crude but works)
        if len(paragraphs) == 1:
            paragraphs = text.replace(". ", ".\n\n").split("\n\n")

        # Build chunks from paragraphs
        chunks_data: list[tuple[str, int]] = []
        current_paragraphs: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self._count_tokens(para)

            # Would adding this paragraph exceed the limit?
            if current_tokens + para_tokens > self.max_tokens and current_paragraphs:
                # Save current chunk
                chunk_text = "\n\n".join(current_paragraphs)
                chunks_data.append((chunk_text, current_tokens))

                # Start new chunk with overlap
                # Include last paragraph of previous chunk for context
                if self.overlap_tokens > 0 and current_paragraphs:
                    last_para = current_paragraphs[-1]
                    if self._count_tokens(last_para) <= self.overlap_tokens:
                        current_paragraphs = [last_para]
                        current_tokens = self._count_tokens(last_para)
                    else:
                        current_paragraphs = []
                        current_tokens = 0
                else:
                    current_paragraphs = []
                    current_tokens = 0

            current_paragraphs.append(para)
            current_tokens += para_tokens

        # Don't forget the last chunk
        if current_paragraphs:
            chunk_text = "\n\n".join(current_paragraphs)
            # Only add if it's substantial enough
            if self._count_tokens(chunk_text) >= self.min_chunk_tokens:
                chunks_data.append((chunk_text, self._count_tokens(chunk_text)))
            elif chunks_data:
                # Append to previous chunk if too small
                prev_text, prev_tokens = chunks_data[-1]
                combined = prev_text + "\n\n" + chunk_text
                chunks_data[-1] = (combined, self._count_tokens(combined))

        # Convert to DocumentChunk objects
        total_chunks = len(chunks_data)
        return [
            self._create_chunk(label, section, text, idx, total_chunks)
            for idx, (text, _) in enumerate(chunks_data)
        ]

    def chunk_drug_label(self, label: DrugLabel) -> list[DocumentChunk]:
        """
        Chunk a drug label into retrieval-optimized pieces.

        Processes each section independently to preserve section boundaries.

        Args:
            label: Parsed drug label

        Returns:
            List of document chunks with metadata
        """
        all_chunks = []

        for section in label.sections:
            section_chunks = self._chunk_section(label, section)
            all_chunks.extend(section_chunks)

            logger.debug(
                "chunked_section",
                drug=label.drug_name,
                section=section.section_type,
                chunks=len(section_chunks),
            )

        logger.debug(
            "chunked_label",
            drug=label.drug_name,
            total_chunks=len(all_chunks),
            sections=len(label.sections),
        )

        return all_chunks

    def chunk_all_labels(self, labels: list[DrugLabel]) -> list[DocumentChunk]:
        """
        Chunk multiple drug labels.

        Args:
            labels: List of parsed drug labels

        Returns:
            List of all document chunks
        """
        all_chunks = []

        for i, label in enumerate(labels):
            chunks = self.chunk_drug_label(label)
            all_chunks.extend(chunks)

            # Progress logging
            if (i + 1) % 100 == 0:
                logger.info(
                    "chunking_progress",
                    completed=i + 1,
                    total=len(labels),
                    chunks_so_far=len(all_chunks),
                )

        logger.info(
            "chunking_complete",
            labels=len(labels),
            total_chunks=len(all_chunks),
            avg_chunks_per_label=len(all_chunks) / len(labels) if labels else 0,
        )

        return all_chunks


def save_chunks_to_jsonl(chunks: list[DocumentChunk], output_path) -> None:
    """
    Save chunks to JSONL format for later use.

    JSONL (JSON Lines) is one JSON object per line, which is:
    - Easy to stream/process
    - Easy to inspect manually
    - Compatible with many tools
    """
    import json
    from pathlib import Path
    from dataclasses import asdict

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for chunk in chunks:
            # Convert dataclass to dict
            chunk_dict = asdict(chunk)
            f.write(json.dumps(chunk_dict) + "\n")

    logger.info("saved_chunks", path=str(output_path), count=len(chunks))


def load_chunks_from_jsonl(input_path) -> list[DocumentChunk]:
    """Load chunks from JSONL file."""
    import json
    from pathlib import Path

    chunks = []
    with open(Path(input_path)) as f:
        for line in f:
            data = json.loads(line)
            chunks.append(DocumentChunk(**data))

    return chunks


if __name__ == "__main__":
    # Quick test: chunk parsed drug labels
    from pathlib import Path
    from drugquery.logging import configure_logging
    from drugquery.ingestion.parse import parse_all_spl_files

    configure_logging()

    input_dir = Path("data/raw")
    xml_files = list(input_dir.glob("*.xml"))

    if not xml_files:
        print("No XML files found. Run download first:")
        print("  uv run python -m drugquery.ingestion.download")
    else:
        # Parse all files
        labels = parse_all_spl_files(input_dir)

        if labels:
            # Chunk all labels
            chunker = DrugLabelChunker()
            chunks = chunker.chunk_all_labels(labels)

            # Save to processed directory
            output_path = Path("data/processed/chunks.jsonl")
            save_chunks_to_jsonl(chunks, output_path)

            # Print summary
            print(f"\n{'='*60}")
            print(f"Processed {len(labels)} drug labels into {len(chunks)} chunks")
            print(f"Saved to: {output_path}")
            print(f"{'='*60}")

            # Show some examples
            print("\nSample chunks:")
            for chunk in chunks[:3]:
                print(f"\n  {chunk.chunk_id}")
                print(f"    Drug: {chunk.drug_name}")
                print(f"    Section: {chunk.section_type}")
                print(f"    Tokens: {chunk.token_count}")
                preview = chunk.content[:80] + "..." if len(chunk.content) > 80 else chunk.content
                print(f"    Content: {preview}")