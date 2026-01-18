"""
Download FDA DailyMed SPL documents.

DailyMed provides drug label data via a REST API:
https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm

Key concepts:
- SPL (Structured Product Labeling): XML format for drug labels
- SetID: Unique identifier for each drug label
- Each drug may have multiple versions; we download the latest

Usage:
    from drugquery.ingestion.download import download_drug_labels

    downloaded = await download_drug_labels(Path("data/raw"), limit=100)
"""

import asyncio
from pathlib import Path

import httpx

from drugquery.logging import get_logger

logger = get_logger(__name__, component="download")

# DailyMed API base URL
DAILYMED_BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2"

# Rate limiting: DailyMed asks for reasonable request rates
# We'll add a small delay between requests to be good citizens
REQUEST_DELAY_SECONDS = 0.1


async def get_available_spls(
    client: httpx.AsyncClient,
    page_size: int = 100,
    max_pages: int = 5,
) -> list[dict]:
    """
    Get list of available SPL documents from DailyMed.

    The API returns paginated results. We fetch multiple pages
    up to our desired limit.

    Args:
        client: HTTP client for making requests
        page_size: Number of results per page (max 100)
        max_pages: Maximum number of pages to fetch

    Returns:
        List of SPL metadata dictionaries containing setid, title, etc.
    """
    all_spls = []

    for page in range(1, max_pages + 1):
        logger.info("fetching_spl_list", page=page, page_size=page_size)

        response = await client.get(
            f"{DAILYMED_BASE_URL}/spls.json",
            params={
                "pagesize": page_size,
                "page": page,
            },
        )
        response.raise_for_status()

        data = response.json()
        spls = data.get("data", [])

        if not spls:
            # No more results
            break

        all_spls.extend(spls)
        logger.info("fetched_spl_page", page=page, count=len(spls))

        # Small delay to be respectful to the API
        await asyncio.sleep(REQUEST_DELAY_SECONDS)

    return all_spls


async def download_spl_xml(
    client: httpx.AsyncClient,
    set_id: str,
    output_dir: Path,
) -> Path | None:
    """
    Download a single SPL XML file.

    Args:
        client: HTTP client for making requests
        set_id: The unique identifier for the SPL document
        output_dir: Directory to save the XML file

    Returns:
        Path to downloaded file, or None if download failed
    """
    output_path = output_dir / f"{set_id}.xml"

    # Skip if already downloaded
    if output_path.exists():
        logger.debug("skipping_existing", set_id=set_id)
        return output_path

    try:
        response = await client.get(
            f"{DAILYMED_BASE_URL}/spls/{set_id}.xml",
        )
        response.raise_for_status()

        # Save to disk
        output_path.write_bytes(response.content)
        logger.debug("downloaded_spl", set_id=set_id, size=len(response.content))

        return output_path

    except httpx.HTTPStatusError as e:
        logger.warning("download_failed_http", set_id=set_id, status=e.response.status_code)
        return None
    except Exception as e:
        logger.warning("download_failed", set_id=set_id, error=str(e))
        return None


async def download_drug_labels(
    output_dir: Path,
    limit: int = 500,
) -> list[Path]:
    """
    Download SPL files from FDA DailyMed.

    This is the main entry point for downloading drug labels.
    It fetches the list of available SPLs, then downloads each one.

    Args:
        output_dir: Directory to save downloaded XML files
        limit: Maximum number of drug labels to download

    Returns:
        List of paths to successfully downloaded XML files

    Example:
        >>> import asyncio
        >>> from pathlib import Path
        >>>
        >>> downloaded = asyncio.run(
        ...     download_drug_labels(Path("data/raw"), limit=100)
        ... )
        >>> print(f"Downloaded {len(downloaded)} files")
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("starting_download", output_dir=str(output_dir), limit=limit)

    # Calculate pagination
    page_size = min(100, limit)  # API max is 100
    max_pages = (limit + page_size - 1) // page_size

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Get list of available SPLs
        available_spls = await get_available_spls(
            client,
            page_size=page_size,
            max_pages=max_pages,
        )

        # Limit to requested number
        spls_to_download = available_spls[:limit]
        logger.info("spls_to_download", count=len(spls_to_download))

        # Step 2: Download each SPL
        downloaded_paths = []

        for i, spl in enumerate(spls_to_download):
            set_id = spl.get("setid")
            if not set_id:
                continue

            # Progress logging every 50 files
            if (i + 1) % 50 == 0:
                logger.info("download_progress", completed=i + 1, total=len(spls_to_download))

            path = await download_spl_xml(client, set_id, output_dir)

            if path:
                downloaded_paths.append(path)

            # Rate limiting
            await asyncio.sleep(REQUEST_DELAY_SECONDS)

    logger.info(
        "download_complete",
        requested=limit,
        downloaded=len(downloaded_paths),
        output_dir=str(output_dir),
    )

    return downloaded_paths


# Convenience function for synchronous usage
def download_drug_labels_sync(output_dir: Path, limit: int = 500) -> list[Path]:
    """
    Synchronous wrapper for download_drug_labels.

    Use this when you're not in an async context.

    Args:
        output_dir: Directory to save downloaded XML files
        limit: Maximum number of drug labels to download

    Returns:
        List of paths to successfully downloaded XML files
    """
    return asyncio.run(download_drug_labels(output_dir, limit))


if __name__ == "__main__":
    # Quick test: download 5 drug labels
    from drugquery.logging import configure_logging

    configure_logging()

    output = Path("data/raw")
    downloaded = download_drug_labels_sync(output, limit=5)

    print(f"\nDownloaded {len(downloaded)} files to {output}")
    for path in downloaded:
        print(f"  - {path.name}")