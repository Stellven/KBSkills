"""Skill executor - applies activated skills to the agent workflow."""

from kbskills.skills.loader import Skill
from kbskills.skills.matcher import SkillMatch


def build_skill_system_prompt(matches: list[SkillMatch]) -> str:
    """Build a system prompt section from activated skills.

    Merges thinking frameworks from all matched skills, ordered by score.
    """
    if not matches:
        return ""

    parts = []
    parts.append("## Activated Thinking Skills\n")

    for match in matches:
        skill = match.skill
        parts.append(f"### {skill.metadata.display_name} (relevance: {match.score:.2f})")
        parts.append(skill.thinking_framework.description)
        parts.append("")

    return "\n".join(parts)


def build_skill_steps_prompt(matches: list[SkillMatch], topic: str) -> str:
    """Build prompts for skill-specific thinking steps.

    Returns a combined prompt with all steps from activated skills.
    """
    if not matches:
        return ""

    parts = []
    parts.append(f"## Skill-Guided Analysis for: {topic}\n")
    parts.append("Apply the following thinking frameworks to analyze this topic:\n")

    for match in matches:
        skill = match.skill
        parts.append(f"### {skill.metadata.display_name}")

        for step in skill.thinking_framework.steps:
            prompt = step.prompt.replace("{topic}", topic)
            parts.append(f"**{step.name}:**")
            parts.append(prompt)
            parts.append("")

    return "\n".join(parts)


def build_output_requirements(matches: list[SkillMatch]) -> dict:
    """Merge output requirements from all activated skills.

    Higher-scoring skills take precedence for conflicting requirements.
    """
    all_sections = []
    style_parts = []

    for match in matches:
        reqs = match.skill.output_requirements
        for section in reqs.sections:
            if section not in all_sections:
                all_sections.append(section)
        if reqs.style:
            style_parts.append(reqs.style)

    return {
        "sections": all_sections,
        "style": " ".join(style_parts),
    }


def build_tools_format(matches: list[SkillMatch]) -> str:
    """Build output format hints from skill tools."""
    if not matches:
        return ""

    parts = []
    for match in matches:
        for tool in match.skill.tools:
            if tool.output_format:
                parts.append(f"**{tool.name}** ({tool.description}):")
                parts.append(tool.output_format)
                parts.append("")

    return "\n".join(parts) if parts else ""


def format_activated_skills_header(matches: list[SkillMatch]) -> str:
    """Format the activated skills line for the outline header."""
    if not matches:
        return ""
    skill_info = ", ".join(
        f"{m.skill.metadata.display_name} ({m.score:.2f})"
        for m in matches
    )
    return f"> 激活 Skills: {skill_info}"
