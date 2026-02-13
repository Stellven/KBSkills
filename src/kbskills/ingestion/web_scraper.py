"""Web scraper - fetches and extracts text content from web pages."""

import re
from urllib.parse import unquote, urlparse

import httpx
import html2text
from bs4 import BeautifulSoup
from rich.console import Console

from kbskills.ingestion.file_loader import Document
from kbskills.utils.text import clean_text

console = Console()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}

MAX_RETRIES = 3
TIMEOUT = 30.0

# Pattern to detect Wikipedia URLs: e.g. https://en.wikipedia.org/wiki/Some_Article
_WIKIPEDIA_PATTERN = re.compile(r"https?://(\w+)\.wikipedia\.org/wiki/(.+)")


def scrape_url(url: str) -> Document | None:
    """Scrape a web page and return its text content as a Document.

    Automatically detects Wikipedia URLs and uses the MediaWiki API for reliable access.
    """
    # Check for Wikipedia — use API instead of HTML scraping
    wiki_match = _WIKIPEDIA_PATTERN.match(url.split("#")[0].split("?")[0])
    if wiki_match:
        return _scrape_wikipedia(url, lang=wiki_match.group(1), title=unquote(wiki_match.group(2)))

    return _scrape_generic(url)


def _scrape_wikipedia(url: str, lang: str, title: str) -> Document | None:
    """Fetch Wikipedia article content via the MediaWiki API."""
    api_url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title.replace("_", " "),
        "prop": "extracts",
        "explaintext": 1,
        "format": "json",
    }
    api_headers = {
        "User-Agent": "KBSkills/0.1 (https://github.com/kbskills; kbskills@users.noreply.github.com)",
        "Api-User-Agent": "KBSkills/0.1 (https://github.com/kbskills; kbskills@users.noreply.github.com)",
    }

    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.get(api_url, params=params, headers=api_headers)
            response.raise_for_status()

        data = response.json()
        pages = data.get("query", {}).get("pages", {})

        for page_id, page in pages.items():
            if page_id == "-1":
                console.print(f"[yellow]Wikipedia article not found: {title}[/yellow]")
                return None
            text = page.get("extract", "")
            if text.strip():
                return Document(
                    source=url,
                    content=clean_text(text),
                    metadata={"type": "web", "title": page.get("title", title), "source_type": "wikipedia"},
                )

        console.print(f"[yellow]No content extracted from Wikipedia: {title}[/yellow]")
        return None

    except Exception as e:
        console.print(f"[red]Wikipedia API error for {title}: {e}[/red]")
        return None


def _scrape_generic(url: str) -> Document | None:
    """Scrape a generic web page using HTTP + html2text."""
    for attempt in range(MAX_RETRIES):
        try:
            with httpx.Client(follow_redirects=True, timeout=TIMEOUT, headers=HEADERS) as client:
                response = client.get(url)
                response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                console.print(f"[yellow]Skipping non-HTML content: {url} ({content_type})[/yellow]")
                return None

            html = response.text
            text = _extract_text(html)

            if not text.strip():
                console.print(f"[yellow]No text content extracted from: {url}[/yellow]")
                return None

            title = _extract_title(html)

            return Document(
                source=url,
                content=clean_text(text),
                metadata={"type": "web", "title": title},
            )

        except httpx.HTTPStatusError as e:
            console.print(f"[yellow]HTTP {e.response.status_code} for {url} (attempt {attempt + 1})[/yellow]")
        except httpx.RequestError as e:
            console.print(f"[yellow]Request error for {url}: {e} (attempt {attempt + 1})[/yellow]")

    console.print(f"[red]Failed to scrape {url} after {MAX_RETRIES} attempts[/red]")
    return None


def _extract_text(html: str) -> str:
    """Extract main text content from HTML using html2text."""
    converter = html2text.HTML2Text()
    converter.ignore_links = False
    converter.ignore_images = True
    converter.ignore_emphasis = False
    converter.body_width = 0  # Don't wrap lines
    return converter.handle(html)


def _extract_title(html: str) -> str:
    """Extract page title from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    return title_tag.get_text(strip=True) if title_tag else ""
