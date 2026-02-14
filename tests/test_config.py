"""Unit tests for kbskills.config module."""

import json
import os

import pytest

from kbskills.config import Config, load_config, save_config


class TestConfigDefaults:
    """Tests for Config dataclass defaults and properties."""

    def test_config_defaults(self):
        config = Config()
        assert config.gemini_api_key == ""
        assert config.data_dir == "./data"
        assert config.output_dir == "./output"
        assert config.skills_dir == "./skills"
        assert config.llm_model == "gemini-2.5-pro"
        assert config.embedding_model == "text-embedding-004"
        assert config.default_search_mode == "hybrid"
        assert config.skill_match_top_k == 3
        assert config.skill_match_default_threshold == 0.6

    def test_config_raw_dir(self):
        config = Config(data_dir="/tmp/test_data")
        assert str(config.raw_dir) == "/tmp/test_data/raw"

    def test_config_graph_dir(self):
        config = Config(data_dir="/tmp/test_data")
        assert str(config.graph_dir) == "/tmp/test_data/graph"

    def test_config_ensure_dirs(self, tmp_path):
        config = Config(
            data_dir=str(tmp_path / "data"),
            output_dir=str(tmp_path / "output"),
            skills_dir=str(tmp_path / "skills"),
        )
        config.ensure_dirs()
        assert config.raw_dir.exists()
        assert config.graph_dir.exists()
        assert (tmp_path / "output").exists()
        assert (tmp_path / "skills").exists()


class TestLoadConfig:
    """Tests for load_config with env var overrides."""

    def test_load_config_from_env(self, monkeypatch):
        monkeypatch.setenv("KBSKILLS_LLM_MODEL", "gemini-2.0-flash")
        monkeypatch.setenv("KBSKILLS_SKILL_MATCH_TOP_K", "5")
        monkeypatch.setenv("KBSKILLS_SKILL_MATCH_DEFAULT_THRESHOLD", "0.8")
        config = load_config()
        assert config.llm_model == "gemini-2.0-flash"
        assert config.skill_match_top_k == 5
        assert config.skill_match_default_threshold == 0.8

    def test_load_config_gemini_key_fallback(self, monkeypatch, tmp_path):
        import kbskills.config as config_mod

        # Point config files to nonexistent paths so real config/env don't interfere
        monkeypatch.setattr(config_mod, "CONFIG_FILE", tmp_path / "missing" / "config.json")
        monkeypatch.chdir(tmp_path)  # No .kbskills.json or .env in tmp_path

        # Clear all potential env sources, then set fallback
        monkeypatch.delenv("KBSKILLS_GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "fallback-key-123")
        config = load_config()
        assert config.gemini_api_key == "fallback-key-123"


class TestSaveConfig:
    """Tests for save_config / load round-trip."""

    def test_save_and_load_config(self, tmp_path, monkeypatch):
        import kbskills.config as config_mod

        config_dir = tmp_path / ".kbskills"
        config_file = config_dir / "config.json"
        monkeypatch.setattr(config_mod, "CONFIG_DIR", config_dir)
        monkeypatch.setattr(config_mod, "CONFIG_FILE", config_file)

        original = Config(
            gemini_api_key="save-test-key",
            llm_model="gemini-2.0-flash",
            skill_match_top_k=7,
        )
        save_config(original)

        assert config_file.exists()
        data = json.loads(config_file.read_text())
        assert data["gemini_api_key"] == "save-test-key"
        assert data["llm_model"] == "gemini-2.0-flash"
        assert data["skill_match_top_k"] == 7
