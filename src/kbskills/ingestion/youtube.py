"""YouTube video transcription."""

import re
from rich.console import Console

from kbskills.ingestion.file_loader import Document
from kbskills.utils.text import clean_text

console = Console()


def extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:shorts/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def transcribe_youtube(url: str) -> Document | None:
    """Get transcript from a YouTube video.

    Tries official transcript API first, falls back to yt-dlp audio download.
    """
    video_id = extract_video_id(url)
    if not video_id:
        console.print(f"[red]Could not extract video ID from: {url}[/red]")
        return None

    # Try transcript API first
    doc = _try_transcript_api(url, video_id)
    if doc:
        return doc

    # Fallback: download audio (handled by audio module)
    console.print(f"[yellow]No transcript available for {video_id}, trying audio download...[/yellow]")
    doc = _try_audio_download(url, video_id)
    return doc


def _try_transcript_api(url: str, video_id: str) -> Document | None:
    """Try to get transcript using youtube-transcript-api."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        ytt_api = YouTubeTranscriptApi()
        # Try to get transcript, preferring Chinese and English
        transcript = ytt_api.fetch(video_id, languages=["zh-Hans", "zh", "en", "zh-Hant"])
        text_parts = [entry.text for entry in transcript.snippets]
        text = " ".join(text_parts)

        if not text.strip():
            return None

        return Document(
            source=url,
            content=clean_text(text),
            metadata={"type": "youtube", "video_id": video_id, "method": "transcript_api"},
        )

    except Exception as e:
        console.print(f"[yellow]Transcript API failed for {video_id}: {e}[/yellow]")
        return None


def _try_audio_download(url: str, video_id: str) -> Document | None:
    """Download audio from YouTube and transcribe it."""
    try:
        import tempfile
        import yt_dlp

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/{video_id}.%(ext)s"
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": output_path,
                "quiet": True,
                "no_warnings": True,
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "128",
                }],
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            # Find the downloaded audio file
            import glob
            audio_files = glob.glob(f"{tmpdir}/{video_id}.*")
            if not audio_files:
                console.print(f"[red]No audio file downloaded for {video_id}[/red]")
                return None

            # Transcribe using audio module
            from kbskills.ingestion.audio import transcribe_audio_file
            text = transcribe_audio_file(audio_files[0])
            if not text:
                return None

            return Document(
                source=url,
                content=clean_text(text),
                metadata={"type": "youtube", "video_id": video_id, "method": "audio_transcription"},
            )

    except Exception as e:
        console.print(f"[red]Audio download failed for {video_id}: {e}[/red]")
        return None
