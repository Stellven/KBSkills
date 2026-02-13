"""Skill YAML file loading and parsing."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SkillTrigger:
    domains: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    intent_patterns: list[str] = field(default_factory=list)
    threshold: float = 0.6


@dataclass
class SkillMetadata:
    name: str
    display_name: str
    version: str = "1.0"
    description: str = ""
    trigger: SkillTrigger = field(default_factory=SkillTrigger)


@dataclass
class ThinkingStep:
    name: str
    prompt: str


@dataclass
class ThinkingFramework:
    description: str = ""
    steps: list[ThinkingStep] = field(default_factory=list)


@dataclass
class SkillTool:
    name: str
    description: str = ""
    output_format: str = ""


@dataclass
class OutputRequirements:
    sections: list[str] = field(default_factory=list)
    style: str = ""


@dataclass
class Skill:
    metadata: SkillMetadata
    thinking_framework: ThinkingFramework = field(default_factory=ThinkingFramework)
    tools: list[SkillTool] = field(default_factory=list)
    output_requirements: OutputRequirements = field(default_factory=OutputRequirements)
    file_path: str = ""


def parse_skill(data: dict, file_path: str = "") -> Skill:
    """Parse a skill from a YAML dictionary."""
    meta_raw = data.get("metadata", {})
    trigger_raw = meta_raw.get("trigger", {})

    trigger = SkillTrigger(
        domains=trigger_raw.get("domains", []),
        keywords=trigger_raw.get("keywords", []),
        intent_patterns=trigger_raw.get("intent_patterns", []),
        threshold=trigger_raw.get("threshold", 0.6),
    )

    metadata = SkillMetadata(
        name=meta_raw.get("name", ""),
        display_name=meta_raw.get("display_name", ""),
        version=meta_raw.get("version", "1.0"),
        description=meta_raw.get("description", ""),
        trigger=trigger,
    )

    # Thinking framework
    tf_raw = data.get("thinking_framework", {})
    steps = []
    for s in tf_raw.get("steps", []):
        steps.append(ThinkingStep(name=s.get("name", ""), prompt=s.get("prompt", "")))

    thinking_framework = ThinkingFramework(
        description=tf_raw.get("description", ""),
        steps=steps,
    )

    # Tools
    tools = []
    for t in data.get("tools", []):
        tools.append(SkillTool(
            name=t.get("name", ""),
            description=t.get("description", ""),
            output_format=t.get("output_format", ""),
        ))

    # Output requirements
    or_raw = data.get("output_requirements", {})
    output_requirements = OutputRequirements(
        sections=or_raw.get("sections", []),
        style=or_raw.get("style", ""),
    )

    return Skill(
        metadata=metadata,
        thinking_framework=thinking_framework,
        tools=tools,
        output_requirements=output_requirements,
        file_path=file_path,
    )


def load_skill_file(path: str | Path) -> Skill:
    """Load a single skill from a YAML file."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    return parse_skill(data, file_path=str(path))


def load_all_skills(skills_dir: str | Path) -> list[Skill]:
    """Load all skill YAML files from the given directory."""
    skills_dir = Path(skills_dir)
    if not skills_dir.exists():
        return []

    skills = []
    for path in sorted(skills_dir.glob("*.yaml")):
        try:
            skill = load_skill_file(path)
            skills.append(skill)
        except Exception as e:
            from rich.console import Console
            Console().print(f"[yellow]Warning: Failed to load skill {path.name}: {e}[/yellow]")

    return skills
