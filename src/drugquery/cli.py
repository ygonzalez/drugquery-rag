"""
Command-line interface for DrugQuery.

Commands:
- download: Download drug labels from FDA DailyMed
- ingest: Parse and chunk drug labels
- index: Index chunks into Weaviate
- search: Search the vector store
- serve: Start the API server
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
@click.option("--input", "input_file", default="data/processed/chunks.jsonl", help="Input chunks file")
@click.option("--reset", is_flag=True, help="Delete existing data before indexing")
def index(input_file: str, reset: bool) -> None:
    """Index chunks into Weaviate vector store."""
    from pathlib import Path
    from drugquery.ingestion.chunk import load_chunks_from_jsonl
    from drugquery.vectorstore import DrugVectorStore

    input_path = Path(input_file)
    if not input_path.exists():
        console.print(f"[red]Chunks file not found: {input_file}[/red]")
        console.print("Run ingestion first: uv run drugquery ingest")
        return

    # Load chunks
    console.print(f"[yellow]Loading chunks from {input_file}...[/yellow]")
    chunks = load_chunks_from_jsonl(input_path)
    console.print(f"[blue]Loaded {len(chunks)} chunks[/blue]")

    # Index into Weaviate
    with DrugVectorStore() as store:
        if reset:
            console.print("[yellow]Resetting collection...[/yellow]")
            store.delete_all()

        stats = store.get_stats()
        console.print(f"[blue]Current documents in store: {stats['count']}[/blue]")

        console.print("[yellow]Indexing chunks (generating embeddings via OpenAI)...[/yellow]")
        indexed = store.index_chunks(chunks)

        stats = store.get_stats()
        console.print(f"[green]✓ Indexed {indexed} chunks. Total: {stats['count']}[/green]")


@main.command()
@click.argument("query")
@click.option("--limit", default=5, help="Number of results")
@click.option("--section", default=None, help="Filter by section type")
@click.option("--drug", default=None, help="Filter by drug name")
def search(query: str, limit: int, section: str | None, drug: str | None) -> None:
    """Search the vector store."""
    from drugquery.vectorstore import DrugVectorStore

    with DrugVectorStore() as store:
        stats = store.get_stats()
        if stats["count"] == 0:
            console.print("[red]No documents indexed. Run: uv run drugquery index[/red]")
            return

        console.print(f"[yellow]Searching for: {query}[/yellow]")

        results = store.hybrid_search(
            query=query,
            limit=limit,
            section_filter=section,
            drug_filter=drug,
        )

        if not results:
            console.print("[yellow]No results found.[/yellow]")
            return

        console.print(f"\n[green]Found {len(results)} results:[/green]\n")

        for i, result in enumerate(results, 1):
            score = result.get("score", 0)
            console.print(f"[bold]--- Result {i} (score: {score:.4f}) ---[/bold]")
            console.print(f"[blue]Drug:[/blue] {result['drug_name']}")
            console.print(f"[blue]Section:[/blue] {result['section_type']}")
            preview = result['content'][:300] + "..." if len(result['content']) > 300 else result['content']
            console.print(f"[blue]Content:[/blue] {preview}")
            console.print()


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
    console.print("[red]Not yet implemented - Phase 5[/red]")


if __name__ == "__main__":
    main()