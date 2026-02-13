"""Audio transcription using Google Gemini API."""

import os
from pathlib import Path

from rich.console import Console

from kbskills.utils.text import clean_text

console = Console()


def transcribe_audio_file(file_path: str, api_key: str | None = None) -> str | None:
    """Transcribe an audio file using Google Gemini API.

    Args:
        file_path: Path to the audio file.
        api_key: Gemini API key. If None, reads from GEMINI_API_KEY env var.

    Returns:
        Transcribed text, or None if failed.
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        console.print("[red]Error: Gemini API key required for audio transcription[/red]")
        return None

    file_path = Path(file_path)
    if not file_path.exists():
        console.print(f"[red]Audio file not found: {file_path}[/red]")
        return None

    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        # Upload the audio file
        console.print(f"[dim]Uploading audio: {file_path.name}...[/dim]")
        uploaded_file = client.files.upload(file=file_path)

        # Transcribe using Gemini
        console.print(f"[dim]Transcribing audio: {file_path.name}...[/dim]")
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[
                uploaded_file,
                "Please transcribe this audio file completely and accurately. "
                "Output only the transcription text, no additional commentary.",
            ],
        )

        text = response.text
        if text:
            return clean_text(text)
        return None

    except Exception as e:
        console.print(f"[red]Audio transcription failed for {file_path}: {e}[/red]")
        return None


def transcribe_audio_url(url: str, api_key: str | None = None) -> str | None:
    """Download and transcribe an audio file from a URL."""
    import tempfile
    import httpx

    try:
        # Download the audio file
        console.print(f"[dim]Downloading audio from: {url}...[/dim]")
        with httpx.Client(follow_redirects=True, timeout=120.0) as client:
            response = client.get(url)
            response.raise_for_status()

        # Determine extension from URL
        path_part = url.split("?")[0]
        ext = Path(path_part).suffix or ".mp3"

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(response.content)
            tmp_path = f.name

        try:
            return transcribe_audio_file(tmp_path, api_key)
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        console.print(f"[red]Failed to download audio from {url}: {e}[/red]")
        return None
