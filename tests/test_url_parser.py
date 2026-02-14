"""Unit tests for kbskills.ingestion.url_parser module."""

import pytest
from pathlib import Path

from kbskills.ingestion.url_parser import classify_url, parse_url_file, URLType


class TestClassifyUrl:

    # ── YouTube ──────────────────────────────────────────────────────────────

    def test_youtube_watch(self):
        assert classify_url("https://www.youtube.com/watch?v=abc123") == URLType.YOUTUBE

    def test_youtube_watch_no_www(self):
        assert classify_url("https://youtube.com/watch?v=abc123") == URLType.YOUTUBE

    def test_youtube_short_url(self):
        assert classify_url("https://youtu.be/abc123") == URLType.YOUTUBE

    def test_youtube_shorts(self):
        assert classify_url("https://www.youtube.com/shorts/abc123") == URLType.YOUTUBE

    def test_youtube_http(self):
        assert classify_url("http://youtube.com/watch?v=xyz") == URLType.YOUTUBE

    # ── Audio ────────────────────────────────────────────────────────────────

    def test_audio_mp3(self):
        assert classify_url("https://example.com/podcast.mp3") == URLType.AUDIO

    def test_audio_wav(self):
        assert classify_url("https://example.com/sound.wav") == URLType.AUDIO

    def test_audio_with_query_params(self):
        assert classify_url("https://cdn.example.com/file.flac?token=abc") == URLType.AUDIO

    def test_audio_m4a(self):
        assert classify_url("https://example.com/talk.m4a") == URLType.AUDIO

    # ── Web (default) ────────────────────────────────────────────────────────

    def test_web_regular(self):
        assert classify_url("https://example.com/article") == URLType.WEB

    def test_web_with_path(self):
        assert classify_url("https://docs.python.org/3/library/re.html") == URLType.WEB

    def test_web_case_insensitive(self):
        assert classify_url("HTTPS://EXAMPLE.COM/PAGE") == URLType.WEB


class TestParseUrlFile:

    def test_parse_url_file(self, tmp_path):
        url_file = tmp_path / "urls.txt"
        url_file.write_text(
            "# This is a comment\n"
            "https://example.com/page\n"
            "\n"
            "https://www.youtube.com/watch?v=abc\n"
            "https://cdn.example.com/audio.mp3\n"
            "# Another comment\n"
        )
        results = parse_url_file(url_file)

        assert len(results) == 3
        assert results[0].url == "https://example.com/page"
        assert results[0].url_type == URLType.WEB
        assert results[1].url_type == URLType.YOUTUBE
        assert results[2].url_type == URLType.AUDIO

    def test_parse_url_file_empty(self, tmp_path):
        url_file = tmp_path / "empty.txt"
        url_file.write_text("")
        assert parse_url_file(url_file) == []

    def test_parse_url_file_comments_only(self, tmp_path):
        url_file = tmp_path / "comments.txt"
        url_file.write_text("# comment 1\n# comment 2\n")
        assert parse_url_file(url_file) == []
