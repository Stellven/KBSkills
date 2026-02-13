"""Skill semantic matching engine."""

import re
from dataclasses import dataclass, field

import numpy as np
from rich.console import Console

from kbskills.config import Config
from kbskills.skills.loader import Skill
from kbskills.utils.retry import retry_embedding_call, EmbeddingError

console = Console()


@dataclass
class SkillMatch:
    skill: Skill
    score: float
    matched_domains: list[str] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)


class SkillMatcher:
    """Matches user topics to relevant skills using semantic similarity."""

    def __init__(self, config: Config):
        self.config = config
        self.top_k = config.skill_match_top_k
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.config.gemini_api_key)
        return self._client

    @retry_embedding_call(max_retries=3, min_wait=2, max_wait=15)
    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for a list of texts.

        Retries up to 3 times with exponential backoff on API errors.
        """
        try:
            result = self.client.models.embed_content(
                model=f"models/{self.config.embedding_model}",
                contents=texts,
            )
            return [e.values for e in result.embeddings]
        except Exception as e:
            raise EmbeddingError(f"Embedding API call failed: {e}") from e

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a_arr = np.array(a)
        b_arr = np.array(b)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))

    def _compute_score(self, topic: str, topic_embedding: list[float], skill: Skill) -> tuple[float, list[str], list[str]]:
        """Compute match score for a skill against a topic.

        Returns (score, matched_domains, matched_keywords).
        """
        trigger = skill.metadata.trigger

        # 1. Domain similarity (0.5 weight)
        domain_sim = 0.0
        matched_domains = []
        if trigger.domains:
            domain_embeddings = self._embed(trigger.domains)
            for domain, d_emb in zip(trigger.domains, domain_embeddings):
                sim = self._cosine_similarity(topic_embedding, d_emb)
                if sim > domain_sim:
                    domain_sim = sim
                if sim > 0.4:
                    matched_domains.append(domain)

        # 2. Keyword matching (0.3 weight)
        keyword_score = 0.0
        matched_keywords = []
        if trigger.keywords:
            topic_lower = topic.lower()
            for kw in trigger.keywords:
                if kw.lower() in topic_lower:
                    matched_keywords.append(kw)
            keyword_score = len(matched_keywords) / len(trigger.keywords) if trigger.keywords else 0

        # 3. Intent pattern matching (0.2 weight)
        intent_score = 0.0
        if trigger.intent_patterns:
            matches = 0
            for pattern in trigger.intent_patterns:
                try:
                    if re.search(pattern, topic):
                        matches += 1
                except re.error:
                    pass
            intent_score = matches / len(trigger.intent_patterns)

        score = 0.5 * domain_sim + 0.3 * keyword_score + 0.2 * intent_score
        return score, matched_domains, matched_keywords

    def match(self, topic: str, skills: list[Skill]) -> list[SkillMatch]:
        """Match a topic against all available skills.

        Returns matched skills sorted by score (descending), filtered by threshold.
        """
        if not skills:
            return []

        # Get topic embedding once
        topic_embedding = self._embed([topic])[0]

        matches = []
        for skill in skills:
            score, matched_domains, matched_keywords = self._compute_score(
                topic, topic_embedding, skill
            )
            threshold = skill.metadata.trigger.threshold
            if score >= threshold:
                matches.append(SkillMatch(
                    skill=skill,
                    score=score,
                    matched_domains=matched_domains,
                    matched_keywords=matched_keywords,
                ))

        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:self.top_k]
