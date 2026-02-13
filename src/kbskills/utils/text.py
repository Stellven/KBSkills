"""Text processing utilities."""

import re


def clean_text(text: str) -> str:
    """Clean text by removing excessive whitespace and normalizing newlines."""
    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove excessive blank lines (keep at most 2)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 4000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks for processing.

    Tries to split at paragraph boundaries first, then sentence boundaries.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunks.append(text[start:])
            break

        # Try to find a paragraph break near the end
        split_pos = text.rfind("\n\n", start + chunk_size // 2, end)
        if split_pos == -1:
            # Try sentence break
            split_pos = text.rfind(". ", start + chunk_size // 2, end)
            if split_pos != -1:
                split_pos += 2  # Include the period and space
        if split_pos == -1:
            split_pos = end

        chunks.append(text[start:split_pos])
        start = split_pos - overlap

    return chunks


def truncate(text: str, max_length: int = 500) -> str:
    """Truncate text to max_length, adding ellipsis if needed."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."
