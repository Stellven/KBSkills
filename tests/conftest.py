"""Shared test fixtures for KBSkills unit tests."""

import pytest
from pathlib import Path

from kbskills.config import Config
from kbskills.skills.loader import (
    Skill,
    SkillMetadata,
    SkillTrigger,
    ThinkingFramework,
    ThinkingStep,
    SkillTool,
    OutputRequirements,
    parse_skill,
)
from kbskills.skills.matcher import SkillMatch


# ── Sample YAML dict ─────────────────────────────────────────────────────────

SAMPLE_SKILL_YAML = {
    "metadata": {
        "name": "test_skill",
        "display_name": "Test Thinking Skill",
        "version": "1.0",
        "description": "A test skill for unit testing",
        "trigger": {
            "domains": ["testing", "quality assurance"],
            "keywords": ["test", "verify", "validate"],
            "intent_patterns": [r"test.*code", r"verify.*output"],
            "threshold": 0.4,
        },
    },
    "thinking_framework": {
        "description": "Apply structured testing methodology",
        "steps": [
            {"name": "Identify Scope", "prompt": "Determine scope of {topic}"},
            {"name": "Design Tests", "prompt": "Design tests for {topic}"},
        ],
    },
    "tools": [
        {
            "name": "test_matrix",
            "description": "Test coverage matrix",
            "output_format": "| Case | Status |\n|------|--------|",
        }
    ],
    "output_requirements": {
        "sections": ["Test Plan", "Results", "Coverage"],
        "style": "Include pass/fail status for each test case",
    },
}


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_config(tmp_path):
    """Config with test defaults pointing to a temporary directory."""
    return Config(
        gemini_api_key="test-api-key-for-unit-tests",
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "output"),
        skills_dir=str(tmp_path / "skills"),
        llm_model="gemini-2.5-pro",
        embedding_model="text-embedding-004",
    )


@pytest.fixture
def sample_skill_data():
    """Raw YAML-like dict for skill parsing tests."""
    return dict(SAMPLE_SKILL_YAML)


@pytest.fixture
def sample_skill():
    """Fully parsed Skill object."""
    return parse_skill(SAMPLE_SKILL_YAML, file_path="/fake/test_skill.yaml")


@pytest.fixture
def sample_skill_match(sample_skill):
    """A SkillMatch wrapping the sample skill."""
    return SkillMatch(
        skill=sample_skill,
        score=0.75,
        matched_domains=["testing"],
        matched_keywords=["test"],
    )


@pytest.fixture
def tmp_skills_dir(tmp_path):
    """Temporary directory containing sample skill YAML files."""
    import yaml

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    # Write a valid skill
    (skills_dir / "alpha.yaml").write_text(yaml.dump(SAMPLE_SKILL_YAML))

    # Write a second valid skill with different name
    second = dict(SAMPLE_SKILL_YAML)
    second["metadata"] = dict(second["metadata"])
    second["metadata"]["name"] = "second_skill"
    second["metadata"]["display_name"] = "Second Skill"
    (skills_dir / "beta.yaml").write_text(yaml.dump(second))

    return skills_dir
