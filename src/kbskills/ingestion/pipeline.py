"""Ingestion pipeline - orchestrates data ingestion from all sources."""

import os
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from kbskills.config import Config
from kbskills.ingestion.file_loader import Document, load_directory
from kbskills.ingestion.url_parser import parse_url_file, URLType
from kbskills.ingestion.web_scraper import scrape_url
from kbskills.ingestion.youtube import transcribe_youtube
from kbskills.ingestion.audio import transcribe_audio_url
from kbskills.utils.retry import retry_api_call, IngestionError

console = Console()


def run_ingestion(config: Config, source_dir: str | None = None, urls_file: str | None = None):
    """Run the full ingestion pipeline.

    1. Load local files from directory
    2. Parse URL file and fetch content by type
    3. Insert all documents into the knowledge graph
    """
    documents: list[Document] = []

    # Phase 1: Local files
    if source_dir:
        console.print(f"\n[bold]Ingesting local files from: {source_dir}[/bold]")
        docs = load_directory(source_dir)
        # Handle image files that need Vision API
        for doc in docs:
            if doc.metadata.get("needs_vision"):
                text = _process_image(doc.source, config.gemini_api_key)
                if text:
                    doc.content = text
                    doc.metadata.pop("needs_vision", None)
                else:
                    continue
            if doc.content.strip():
                documents.append(doc)

    # Phase 2: URLs
    if urls_file:
        console.print(f"\n[bold]Processing URLs from: {urls_file}[/bold]")
        parsed_urls = parse_url_file(urls_file)
        console.print(f"Found {len(parsed_urls)} URLs "
                      f"(web: {sum(1 for u in parsed_urls if u.url_type == URLType.WEB)}, "
                      f"youtube: {sum(1 for u in parsed_urls if u.url_type == URLType.YOUTUBE)}, "
                      f"audio: {sum(1 for u in parsed_urls if u.url_type == URLType.AUDIO)})")

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                      console=console) as progress:
            task = progress.add_task("Processing URLs...", total=len(parsed_urls))

            for parsed in parsed_urls:
                progress.update(task, description=f"Processing: {parsed.url[:60]}...")
                doc = None

                if parsed.url_type == URLType.WEB:
                    doc = scrape_url(parsed.url)
                elif parsed.url_type == URLType.YOUTUBE:
                    doc = transcribe_youtube(parsed.url)
                elif parsed.url_type == URLType.AUDIO:
                    text = transcribe_audio_url(parsed.url, config.gemini_api_key)
                    if text:
                        doc = Document(
                            source=parsed.url,
                            content=text,
                            metadata={"type": "audio"},
                        )

                if doc and doc.content.strip():
                    documents.append(doc)
                progress.advance(task)

    if not documents:
        console.print("[yellow]No documents were successfully ingested.[/yellow]")
        return

    console.print(f"\n[bold]Total documents to index: {len(documents)}[/bold]")

    # Phase 3: Insert into knowledge graph
    _insert_into_graph(config, documents)


@retry_api_call(operation_name="VisionAPI", max_retries=3, min_wait=2, max_wait=20)
def _process_image(file_path: str, api_key: str) -> str | None:
    """Extract text description from an image using Gemini Vision.

    Retries up to 3 times with exponential backoff on API errors.
    """
    if not api_key:
        return None
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        uploaded = client.files.upload(file=Path(file_path))
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[
                uploaded,
                "Describe this image in detail, extracting any visible text. "
                "Focus on informational content.",
            ],
        )
        return response.text
    except Exception as e:
        raise IngestionError(f"Image processing failed for {file_path}: {e}") from e


def _insert_into_graph(config: Config, documents: list[Document]):
    """Insert documents into the LightRAG knowledge graph."""
    from kbskills.knowledge.graph_builder import get_rag_instance
    from kbskills.utils.retry import retry_api_call

    console.print("\n[bold]Building knowledge graph...[/bold]")
    rag = get_rag_instance(config)

    @retry_api_call(operation_name="GraphInsert", max_retries=3, min_wait=2, max_wait=20)
    def _insert_single(text: str):
        """Insert a single document with retry."""
        try:
            rag.insert(text)
        except Exception as e:
            raise IngestionError(f"Graph insertion failed: {e}") from e

    failed_count = 0
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  console=console) as progress:
        task = progress.add_task("Inserting documents...", total=len(documents))

        for doc in documents:
            progress.update(task, description=f"Indexing: {doc.source[:50]}...")
            try:
                # Prepend source metadata for context
                text_with_source = f"[Source: {doc.source}]\n\n{doc.content}"
                _insert_single(text_with_source)
            except (IngestionError, Exception) as e:
                failed_count += 1
                console.print(f"[yellow]Failed to index {doc.source} after retries: {e}[/yellow]")
            progress.advance(task)

    if failed_count:
        console.print(f"[yellow]Knowledge graph updated with {failed_count} failed document(s).[/yellow]")
    else:
        console.print("[green]Knowledge graph updated successfully![/green]")
