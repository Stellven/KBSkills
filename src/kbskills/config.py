"""Configuration management for KBSkills."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


CONFIG_DIR = Path.home() / ".kbskills"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_CONFIG = {
    "gemini_api_key": "",
    "data_dir": "./data",
    "output_dir": "./output",
    "skills_dir": "./skills",
    "llm_model": "gemini-2.5-pro",
    "embedding_model": "gemini-embedding-001",
    "default_search_mode": "hybrid",
    "skill_match_top_k": 3,
    "skill_match_default_threshold": 0.35,
}


@dataclass
class Config:
    gemini_api_key: str = ""
    data_dir: str = "./data"
    output_dir: str = "./output"
    skills_dir: str = "./skills"
    llm_model: str = "gemini-2.5-pro"
    embedding_model: str = "text-embedding-004"
    default_search_mode: str = "hybrid"
    skill_match_top_k: int = 3
    skill_match_default_threshold: float = 0.6

    @property
    def raw_dir(self) -> Path:
        return Path(self.data_dir) / "raw"

    @property
    def graph_dir(self) -> Path:
        return Path(self.data_dir) / "graph"

    def ensure_dirs(self):
        """Create necessary directories if they don't exist."""
        for d in [self.raw_dir, self.graph_dir, Path(self.output_dir), Path(self.skills_dir)]:
            d.mkdir(parents=True, exist_ok=True)


def load_config() -> Config:
    """Load configuration from config file and environment variables."""
    # Load .env file from project directory if it exists
    from dotenv import load_dotenv
    load_dotenv()

    data = dict(DEFAULT_CONFIG)

    # Load from config file
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            data.update(json.load(f))

    # Load from project-local JSON config
    local_config = Path(".kbskills.json")
    if local_config.exists():
        with open(local_config) as f:
            data.update(json.load(f))

    # Environment variables override (KBSKILLS_GEMINI_API_KEY, etc.)
    for key in DEFAULT_CONFIG:
        env_key = f"KBSKILLS_{key.upper()}"
        env_val = os.environ.get(env_key)
        if env_val is not None:
            # Convert types for numeric fields
            if key in ("skill_match_top_k",):
                env_val = int(env_val)
            elif key in ("skill_match_default_threshold",):
                env_val = float(env_val)
            data[key] = env_val

    # Also check GEMINI_API_KEY directly
    if not data.get("gemini_api_key"):
        data["gemini_api_key"] = os.environ.get("GEMINI_API_KEY", "")

    return Config(**{k: v for k, v in data.items() if k in Config.__dataclass_fields__})


def save_config(config: Config):
    """Save configuration to the global config file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        k: getattr(config, k)
        for k in Config.__dataclass_fields__
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
