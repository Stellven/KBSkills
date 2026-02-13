"""Prompt templates for the Topic Agent."""

TOPIC_DECOMPOSITION = """Given the topic: "{topic}"

Break it down into 3-5 specific sub-topics or aspects that would be relevant for a knowledge base search.
For each sub-topic, provide a search query optimized for semantic retrieval.

{skill_context}

Output as JSON array (no markdown fencing):
[{{"sub_topic": "...", "query": "..."}}]"""

CONCERN_IDENTIFICATION = """You are analyzing knowledge retrieved from a personal knowledge base on the topic: "{topic}"

Retrieved knowledge:
{retrieved_context}

{skill_steps}

Tasks:
1. Identify the top concerns/focus areas that this knowledge base emphasizes regarding this topic. Look for:
   - Themes that appear repeatedly across multiple documents
   - Perspectives that the knowledge base uniquely highlights
   - Core concepts and their relationships

2. For each concern, explain:
   - WHY the knowledge base focuses on this (evidence from sources)
   - The logical reasoning chain that connects the concern to the topic
   - Supporting evidence: cite specific documents or entities

3. Rank concerns by importance (frequency × relevance)

Output as JSON array (no markdown fencing):
[{{
  "concern": "...",
  "importance": 1-10,
  "reasoning": "Why this matters...",
  "evidence": ["source1", "source2"],
  "logic_chain": "A leads to B because..."
}}]"""

OUTLINE_GENERATION = """Based on the following concern analysis from the knowledge base,
generate a detailed structured outline for the topic: "{topic}"

Concern analysis:
{concern_analysis}

Retrieved knowledge:
{retrieved_context}

{skill_output_requirements}

{tools_format}

Requirements:
- Use hierarchical markdown headings (##, ###, ####)
- Each major section corresponds to a key concern of the knowledge base
- Under each concern heading, include:
  - 「关切说明」: 1-2 sentences on what this concern is about
  - 「关注理由」: Why the knowledge base emphasizes this — cite evidence
  - 「逻辑阐述」: The logical chain explaining how this concern relates to the topic and why it matters
  - 「知识要点」: Key knowledge points from the KB supporting this concern
- Order sections by importance/relevance
- End with a summary section linking all concerns together
- Write in the same language as the topic (Chinese topic → Chinese outline, English topic → English outline)

Output ONLY the markdown outline, no additional commentary."""
