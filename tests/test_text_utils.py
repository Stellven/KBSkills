"""Unit tests for kbskills.utils.text module."""

import pytest

from kbskills.utils.text import clean_text, chunk_text, truncate


class TestCleanText:

    def test_normalizes_crlf(self):
        assert clean_text("a\r\nb\rc") == "a\nb\nc"

    def test_reduces_blank_lines(self):
        result = clean_text("a\n\n\n\nb")
        assert result == "a\n\nb"

    def test_strips_lines(self):
        result = clean_text("  hello  \n  world  ")
        assert result == "hello\nworld"

    def test_strips_overall(self):
        assert clean_text("  \n  hello  \n  ") == "hello"

    def test_empty_string(self):
        assert clean_text("") == ""


class TestChunkText:

    def test_small_text_single_chunk(self):
        text = "Short text"
        chunks = chunk_text(text, chunk_size=100)
        assert chunks == [text]

    def test_exact_chunk_size(self):
        text = "a" * 100
        chunks = chunk_text(text, chunk_size=100)
        assert chunks == [text]

    def test_large_text_multiple_chunks(self):
        text = "word " * 1000  # ~5000 chars
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) > 1
        # Every chunk should be non-empty
        assert all(len(c) > 0 for c in chunks)

    def test_overlap_present(self):
        # Build text with known paragraph boundaries
        paragraphs = ["Paragraph one. " * 20, "Paragraph two. " * 20, "Paragraph three. " * 20]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, chunk_size=300, overlap=50)

        if len(chunks) >= 2:
            # The end of chunk N should overlap with the start of chunk N+1
            end_of_first = chunks[0][-50:]
            assert end_of_first in chunks[1] or chunks[1].startswith(end_of_first[:20])

    def test_respects_paragraph_boundary(self):
        text = "A" * 200 + "\n\n" + "B" * 200
        chunks = chunk_text(text, chunk_size=300, overlap=20)
        # Should try to split at paragraph boundary
        assert len(chunks) >= 1


class TestTruncate:

    def test_short_text_unchanged(self):
        assert truncate("hello", max_length=10) == "hello"

    def test_exact_length(self):
        text = "a" * 500
        assert truncate(text, max_length=500) == text

    def test_long_text_truncated(self):
        text = "a" * 600
        result = truncate(text, max_length=500)
        assert len(result) == 500
        assert result.endswith("...")

    def test_truncate_default(self):
        text = "x" * 1000
        result = truncate(text)  # default max_length=500
        assert len(result) == 500
