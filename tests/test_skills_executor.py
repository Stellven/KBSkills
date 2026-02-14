"""Unit tests for kbskills.skills.executor module."""

import pytest

from kbskills.skills.executor import (
    build_skill_system_prompt,
    build_skill_steps_prompt,
    build_output_requirements,
    build_tools_format,
    format_activated_skills_header,
)
from kbskills.skills.matcher import SkillMatch
from kbskills.skills.loader import (
    Skill,
    SkillMetadata,
    SkillTrigger,
    ThinkingFramework,
    ThinkingStep,
    SkillTool,
    OutputRequirements,
)


def _make_match(
    name="skill",
    display_name="Skill Display",
    score=0.75,
    steps=None,
    tools=None,
    sections=None,
    style="",
):
    """Helper to create a SkillMatch with configurable fields."""
    trigger = SkillTrigger()
    meta = SkillMetadata(name=name, display_name=display_name, trigger=trigger)
    framework = ThinkingFramework(
        description=f"{display_name} framework",
        steps=steps or [],
    )
    skill = Skill(
        metadata=meta,
        thinking_framework=framework,
        tools=tools or [],
        output_requirements=OutputRequirements(sections=sections or [], style=style),
    )
    return SkillMatch(skill=skill, score=score)


class TestBuildSkillSystemPrompt:

    def test_empty_matches(self):
        assert build_skill_system_prompt([]) == ""

    def test_contains_display_name_and_score(self):
        match = _make_match(display_name="First Principles", score=0.82)
        result = build_skill_system_prompt([match])
        assert "First Principles" in result
        assert "0.82" in result

    def test_multiple_matches(self):
        m1 = _make_match(display_name="Alpha", score=0.9)
        m2 = _make_match(display_name="Beta", score=0.6)
        result = build_skill_system_prompt([m1, m2])
        assert "Alpha" in result
        assert "Beta" in result


class TestBuildSkillStepsPrompt:

    def test_empty_matches(self):
        assert build_skill_steps_prompt([], "topic") == ""

    def test_replaces_topic(self):
        steps = [ThinkingStep(name="Step1", prompt="Analyze {topic} deeply.")]
        match = _make_match(steps=steps)
        result = build_skill_steps_prompt([match], "machine learning")
        assert "machine learning" in result
        assert "{topic}" not in result

    def test_contains_step_names(self):
        steps = [
            ThinkingStep(name="Identify", prompt="Do {topic}"),
            ThinkingStep(name="Validate", prompt="Check {topic}"),
        ]
        match = _make_match(steps=steps)
        result = build_skill_steps_prompt([match], "test")
        assert "Identify" in result
        assert "Validate" in result


class TestBuildOutputRequirements:

    def test_merge_sections_no_duplicates(self):
        m1 = _make_match(sections=["Plan", "Results"])
        m2 = _make_match(sections=["Results", "Coverage"])
        reqs = build_output_requirements([m1, m2])
        assert reqs["sections"] == ["Plan", "Results", "Coverage"]

    def test_merge_styles(self):
        m1 = _make_match(style="Be concise.")
        m2 = _make_match(style="Use tables.")
        reqs = build_output_requirements([m1, m2])
        assert "Be concise." in reqs["style"]
        assert "Use tables." in reqs["style"]


class TestBuildToolsFormat:

    def test_empty_matches(self):
        assert build_tools_format([]) == ""

    def test_includes_tool_output_format(self):
        tools = [SkillTool(name="matrix", description="Matrix tool", output_format="| Col1 | Col2 |")]
        match = _make_match(tools=tools)
        result = build_tools_format([match])
        assert "matrix" in result
        assert "| Col1 | Col2 |" in result

    def test_skips_tools_without_format(self):
        tools = [SkillTool(name="no_format", description="No format")]
        match = _make_match(tools=tools)
        result = build_tools_format([match])
        assert result == ""


class TestFormatActivatedSkillsHeader:

    def test_empty_matches(self):
        assert format_activated_skills_header([]) == ""

    def test_contains_skill_names_and_scores(self):
        m1 = _make_match(display_name="FP思考", score=0.85)
        m2 = _make_match(display_name="SWOT", score=0.70)
        result = format_activated_skills_header([m1, m2])
        assert "FP思考" in result
        assert "0.85" in result
        assert "SWOT" in result
        assert "激活 Skills" in result
