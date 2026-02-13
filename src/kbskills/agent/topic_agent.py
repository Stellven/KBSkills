"""Topic Agent - decomposes topics, matches skills, retrieves knowledge, generates outlines."""

import json
import re
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from kbskills.config import Config
from kbskills.agent.prompts import TOPIC_DECOMPOSITION, CONCERN_IDENTIFICATION, OUTLINE_GENERATION
from kbskills.knowledge.graph_builder import query_knowledge
from kbskills.skills.loader import load_all_skills
from kbskills.skills.matcher import SkillMatcher, SkillMatch
from kbskills.skills.executor import (
    build_skill_system_prompt,
    build_skill_steps_prompt,
    build_output_requirements,
    build_tools_format,
    format_activated_skills_header,
)
from kbskills.utils.retry import retry_llm_call, LLMError

console = Console()


class TopicAgent:
    """Agent that processes a topic through the full pipeline."""

    def __init__(self, config: Config):
        self.config = config
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.config.gemini_api_key)
        return self._client

    @retry_llm_call(max_retries=3, min_wait=2, max_wait=30)
    def _llm_call(self, prompt: str) -> str:
        """Make a call to the Gemini LLM.

        Retries up to 3 times with exponential backoff on API errors.
        Raises LLMError if all retries are exhausted.
        """
        try:
            response = self.client.models.generate_content(
                model=self.config.llm_model,
                contents=prompt,
            )
            if response.text is None:
                raise LLMError("LLM returned empty response")
            return response.text
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"Gemini API call failed: {e}") from e

    def run(self, topic: str, search_mode: str = "hybrid", output_path: str | None = None) -> str:
        """Run the full topic agent pipeline.

        Returns the path to the generated outline file.
        """
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                      console=console) as progress:

            # Step 1a: Topic decomposition
            task = progress.add_task("Step 1: Decomposing topic...", total=None)
            sub_topics = self._decompose_topic(topic)
            progress.update(task, description=f"Step 1: Decomposed into {len(sub_topics)} sub-topics")
            progress.stop_task(task)

            # Step 1b: Skill matching
            task = progress.add_task("Step 1b: Matching skills...", total=None)
            skill_matches = self._match_skills(topic)
            if skill_matches:
                names = ", ".join(m.skill.metadata.display_name for m in skill_matches)
                progress.update(task, description=f"Step 1b: Activated skills: {names}")
            else:
                progress.update(task, description="Step 1b: No skills matched")
            progress.stop_task(task)

            # Step 2: Knowledge retrieval
            task = progress.add_task("Step 2: Retrieving knowledge...", total=None)
            retrieved_context = self._retrieve_knowledge(sub_topics, search_mode)
            progress.update(task, description="Step 2: Knowledge retrieved")
            progress.stop_task(task)

            # Step 3: Skill-guided analysis (if skills activated)
            if skill_matches:
                task = progress.add_task("Step 3: Applying skill frameworks...", total=None)
                skill_analysis = self._apply_skill_analysis(topic, retrieved_context, skill_matches)
                retrieved_context = f"{retrieved_context}\n\n--- Skill Analysis ---\n{skill_analysis}"
                progress.update(task, description="Step 3: Skill analysis complete")
                progress.stop_task(task)

            # Step 4: Concern identification
            task = progress.add_task("Step 4: Identifying concerns...", total=None)
            concerns = self._identify_concerns(topic, retrieved_context, skill_matches)
            progress.update(task, description=f"Step 4: Identified {len(concerns)} concerns")
            progress.stop_task(task)

            # Step 5: Outline generation
            task = progress.add_task("Step 5: Generating outline...", total=None)
            outline = self._generate_outline(topic, concerns, retrieved_context, skill_matches)
            progress.update(task, description="Step 5: Outline generated")
            progress.stop_task(task)

        # Add header
        header = self._build_header(topic, skill_matches)
        full_outline = f"{header}\n\n{outline}"

        # Save to file
        output_file = self._save_outline(topic, full_outline, output_path)
        return output_file

    def _decompose_topic(self, topic: str) -> list[dict]:
        """Decompose a topic into sub-topics with search queries."""
        prompt = TOPIC_DECOMPOSITION.format(topic=topic, skill_context="")
        response = self._llm_call(prompt)

        try:
            # Try to parse JSON from response (handle markdown fencing)
            json_str = _extract_json(response)
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            # Fallback: use topic as single query
            console.print("[yellow]Warning: Could not parse sub-topics, using original topic[/yellow]")
            return [{"sub_topic": topic, "query": topic}]

    def _match_skills(self, topic: str) -> list[SkillMatch]:
        """Match topic against available skills."""
        skills = load_all_skills(self.config.skills_dir)
        if not skills:
            return []

        matcher = SkillMatcher(self.config)
        return matcher.match(topic, skills)

    def _retrieve_knowledge(self, sub_topics: list[dict], search_mode: str) -> str:
        """Retrieve knowledge from the graph for all sub-topics."""
        results = []
        for st in sub_topics:
            query = st.get("query", st.get("sub_topic", ""))
            try:
                result = query_knowledge(self.config, query, mode=search_mode)
                if result and result.strip():
                    results.append(f"### {st.get('sub_topic', query)}\n{result}")
            except Exception as e:
                console.print(f"[yellow]Warning: Query failed for '{query}': {e}[/yellow]")

        return "\n\n".join(results) if results else "No relevant knowledge found in the knowledge base."

    def _apply_skill_analysis(self, topic: str, context: str, matches: list[SkillMatch]) -> str:
        """Apply skill-specific thinking steps."""
        steps_prompt = build_skill_steps_prompt(matches, topic)
        prompt = f"""You are analyzing the topic "{topic}" using specific thinking frameworks.

Knowledge context:
{context[:8000]}

{steps_prompt}

Provide your analysis following the frameworks above. Be specific and reference the knowledge context."""

        return self._llm_call(prompt)

    def _identify_concerns(self, topic: str, context: str, matches: list[SkillMatch]) -> list[dict]:
        """Identify key concerns from the retrieved knowledge."""
        skill_steps = build_skill_steps_prompt(matches, topic) if matches else ""

        prompt = CONCERN_IDENTIFICATION.format(
            topic=topic,
            retrieved_context=context[:12000],
            skill_steps=skill_steps,
        )
        response = self._llm_call(prompt)

        try:
            json_str = _extract_json(response)
            concerns = json.loads(json_str)
            # Sort by importance
            concerns.sort(key=lambda c: c.get("importance", 0), reverse=True)
            return concerns
        except (json.JSONDecodeError, ValueError):
            console.print("[yellow]Warning: Could not parse concerns, using raw response[/yellow]")
            return [{"concern": "General Analysis", "importance": 5,
                     "reasoning": response[:500], "evidence": [], "logic_chain": ""}]

    def _generate_outline(self, topic: str, concerns: list[dict],
                          context: str, matches: list[SkillMatch]) -> str:
        """Generate the final markdown outline."""
        output_reqs = build_output_requirements(matches) if matches else {}
        tools_fmt = build_tools_format(matches) if matches else ""

        skill_output_str = ""
        if output_reqs:
            sections = output_reqs.get("sections", [])
            style = output_reqs.get("style", "")
            if sections:
                skill_output_str += "Additional sections to include (from activated skills):\n"
                skill_output_str += "\n".join(f"- {s}" for s in sections)
            if style:
                skill_output_str += f"\nStyle requirement: {style}"

        prompt = OUTLINE_GENERATION.format(
            topic=topic,
            concern_analysis=json.dumps(concerns, ensure_ascii=False, indent=2),
            retrieved_context=context[:12000],
            skill_output_requirements=skill_output_str,
            tools_format=tools_fmt,
        )

        return self._llm_call(prompt)

    def _build_header(self, topic: str, matches: list[SkillMatch]) -> str:
        """Build the outline header."""
        lines = [
            f"# Topic: {topic}",
            "",
            "> 本 Outline 基于知识库检索生成，反映知识库在该主题下的核心关切与洞察。",
        ]
        skills_line = format_activated_skills_header(matches)
        if skills_line:
            lines.append(skills_line)
        return "\n".join(lines)

    def _save_outline(self, topic: str, content: str, output_path: str | None) -> str:
        """Save the outline to a file."""
        if output_path:
            path = Path(output_path)
        else:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            # Generate filename from topic
            safe_name = re.sub(r'[^\w\u4e00-\u9fff-]', '_', topic)[:50]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = output_dir / f"{safe_name}_{timestamp}.md"

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path)


def _extract_json(text: str) -> str:
    """Extract JSON from a response that may contain markdown fencing."""
    # Try to find JSON array in the text
    text = text.strip()

    # Remove markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines if they are fences
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # Find the first [ and last ]
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]

    return text
