"""
Command-line interface for DrugQuery.

Provides commands for:
- Data download and ingestion
- Running evaluation
- Starting the API server
"""

import click
from rich.console import Console

from drugquery.logging import configure_logging

console = Console()


@click.group()
@click.version_option()
def main() -> None:
    """DrugQuery RAG - FDA drug information Q&A system."""
    configure_logging()


@main.command()
@click.option("--limit", default=500, help="Number of drug labels to download")
@click.option("--output", default="data/raw", help="Output directory")
def download(limit: int, output: str) -> None:
    """Download drug labels from FDA DailyMed."""
    from pathlib import Path
    from drugquery.ingestion.download import download_drug_labels_sync

    console.print(f"[yellow]Downloading {limit} drug labels to {output}...[/yellow]")

    output_path = Path(output)
    downloaded = download_drug_labels_sync(output_path, limit=limit)

    console.print(f"[green]✓ Downloaded {len(downloaded)} files to {output}[/green]")


@main.command()
@click.option("--input", "input_dir", default="data/raw", help="Input directory with XML files")
@click.option("--output", default="data/processed/chunks.jsonl", help="Output file for chunks")
def ingest(input_dir: str, output: str) -> None:
    """Parse and chunk drug labels."""
    from pathlib import Path
    from drugquery.ingestion.parse import parse_all_spl_files
    from drugquery.ingestion.chunk import DrugLabelChunker, save_chunks_to_jsonl

    console.print(f"[yellow]Processing files from {input_dir}...[/yellow]")

    # Parse XML files
    labels = parse_all_spl_files(Path(input_dir))
    console.print(f"[blue]Parsed {len(labels)} drug labels[/blue]")

    # Chunk
    chunker = DrugLabelChunker()
    chunks = chunker.chunk_all_labels(labels)
    console.print(f"[blue]Created {len(chunks)} chunks[/blue]")

    # Save
    save_chunks_to_jsonl(chunks, output)
    console.print(f"[green]✓ Saved chunks to {output}[/green]")


@main.command()
@click.option("--host", default="0.0.0.0", help="API host")
@click.option("--port", default=8000, help="API port")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def serve(host: str, port: int, reload: bool) -> None:
    """Start the API server."""
    import uvicorn

    uvicorn.run(
        "drugquery.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


@main.command()
@click.option("--test-set", default="data/evaluation/test_set.json", help="Test set path")
def evaluate(test_set: str) -> None:
    """Run evaluation pipeline."""
    console.print(f"[yellow]Running evaluation with {test_set}...[/yellow]")
    # TODO: Implement
    console.print("[red]Not yet implemented - Phase 5[/red]")


if __name__ == "__main__":
    main()