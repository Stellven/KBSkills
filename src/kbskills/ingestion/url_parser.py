"""URL parser - reads a text file of URLs and classifies them by type."""

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class URLType(Enum):
    WEB = "web"
    YOUTUBE = "youtube"
    AUDIO = "audio"


YOUTUBE_PATTERNS = [
    re.compile(r"https?://(www\.)?youtube\.com/watch"),
    re.compile(r"https?://youtu\.be/"),
    re.compile(r"https?://(www\.)?youtube\.com/shorts/"),
]

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}


@dataclass
class ParsedURL:
    url: str
    url_type: URLType


def classify_url(url: str) -> URLType:
    """Classify a URL as web, youtube, or audio."""
    url_lower = url.lower().strip()

    for pattern in YOUTUBE_PATTERNS:
        if pattern.match(url_lower):
            return URLType.YOUTUBE

    # Check if URL ends with an audio extension (ignore query params)
    path_part = url_lower.split("?")[0]
    for ext in AUDIO_EXTENSIONS:
        if path_part.endswith(ext):
            return URLType.AUDIO

    return URLType.WEB


def parse_url_file(file_path: str | Path) -> list[ParsedURL]:
    """Parse a text file containing URLs (one per line) and classify each."""
    results = []
    with open(file_path) as f:
        for line in f:
            url = line.strip()
            if not url or url.startswith("#"):
                continue
            results.append(ParsedURL(url=url, url_type=classify_url(url)))
    return results
