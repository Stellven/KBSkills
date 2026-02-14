"""Unit tests for kbskills.skills.loader module."""

import pytest
import yaml
from pathlib import Path

from kbskills.skills.loader import (
    Skill,
    SkillMetadata,
    SkillTrigger,
    ThinkingFramework,
    ThinkingStep,
    SkillTool,
    OutputRequirements,
    parse_skill,
    load_skill_file,
    load_all_skills,
)


class TestParseSkill:
    """Tests for parse_skill function."""

    def test_parse_skill_full(self, sample_skill_data):
        skill = parse_skill(sample_skill_data, file_path="/fake/skill.yaml")

        assert skill.metadata.name == "test_skill"
        assert skill.metadata.display_name == "Test Thinking Skill"
        assert skill.metadata.version == "1.0"
        assert skill.metadata.description == "A test skill for unit testing"
        assert skill.file_path == "/fake/skill.yaml"

        # Trigger
        trigger = skill.metadata.trigger
        assert trigger.domains == ["testing", "quality assurance"]
        assert trigger.keywords == ["test", "verify", "validate"]
        assert len(trigger.intent_patterns) == 2
        assert trigger.threshold == 0.4

        # Thinking framework
        assert "testing methodology" in skill.thinking_framework.description
        assert len(skill.thinking_framework.steps) == 2
        assert skill.thinking_framework.steps[0].name == "Identify Scope"

        # Tools
        assert len(skill.tools) == 1
        assert skill.tools[0].name == "test_matrix"

        # Output requirements
        assert "Test Plan" in skill.output_requirements.sections
        assert len(skill.output_requirements.sections) == 3

    def test_parse_skill_minimal(self):
        data = {"metadata": {"name": "bare", "display_name": "Bare Skill"}}
        skill = parse_skill(data)

        assert skill.metadata.name == "bare"
        assert skill.metadata.trigger.domains == []
        assert skill.thinking_framework.steps == []
        assert skill.tools == []
        assert skill.output_requirements.sections == []

    def test_parse_skill_trigger_defaults(self):
        data = {"metadata": {"name": "x", "display_name": "X"}}
        skill = parse_skill(data)
        assert skill.metadata.trigger.threshold == 0.6

    def test_parse_skill_empty_dict(self):
        skill = parse_skill({})
        assert skill.metadata.name == ""
        assert skill.metadata.display_name == ""


class TestLoadSkillFile:
    """Tests for load_skill_file."""

    def test_load_skill_file(self, tmp_skills_dir):
        skill = load_skill_file(tmp_skills_dir / "alpha.yaml")
        assert skill.metadata.name == "test_skill"
        assert "alpha.yaml" in skill.file_path


class TestLoadAllSkills:
    """Tests for load_all_skills."""

    def test_load_all_skills(self, tmp_skills_dir):
        skills = load_all_skills(tmp_skills_dir)
        assert len(skills) == 2
        # Sorted alphabetically by filename
        names = [s.metadata.name for s in skills]
        assert names[0] == "test_skill"  # alpha.yaml
        assert names[1] == "second_skill"  # beta.yaml

    def test_load_all_skills_empty_dir(self, tmp_path):
        empty_dir = tmp_path / "empty_skills"
        empty_dir.mkdir()
        assert load_all_skills(empty_dir) == []

    def test_load_all_skills_nonexistent_dir(self, tmp_path):
        assert load_all_skills(tmp_path / "nonexistent") == []

    def test_load_all_skills_invalid_yaml(self, tmp_path):
        skills_dir = tmp_path / "bad_skills"
        skills_dir.mkdir()
        # Use content that causes parse_skill to fail (metadata key missing → KeyError etc.)
        # yaml.safe_load(None) would fail, but we need a file that loads but breaks parse.
        # Actually, write truly invalid YAML using tab indentation:
        (skills_dir / "bad.yaml").write_text("metadata:\n\t- broken: [unclosed")
        # Should not raise, just skip the bad file
        skills = load_all_skills(skills_dir)
        assert len(skills) == 0
