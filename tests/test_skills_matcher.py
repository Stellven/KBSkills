"""Unit tests for kbskills.skills.matcher module."""

import pytest
from unittest.mock import patch, MagicMock

import numpy as np

from kbskills.config import Config
from kbskills.skills.loader import Skill, SkillMetadata, SkillTrigger, ThinkingFramework
from kbskills.skills.matcher import SkillMatcher, SkillMatch


def _make_skill(
    name="skill",
    domains=None,
    keywords=None,
    intent_patterns=None,
    threshold=0.4,
):
    """Helper to build a Skill with minimal boilerplate."""
    trigger = SkillTrigger(
        domains=domains or [],
        keywords=keywords or [],
        intent_patterns=intent_patterns or [],
        threshold=threshold,
    )
    meta = SkillMetadata(name=name, display_name=name, trigger=trigger)
    return Skill(metadata=meta)


class TestCosineSimilarity:
    """Tests for SkillMatcher._cosine_similarity."""

    def setup_method(self):
        self.matcher = SkillMatcher(Config(gemini_api_key="fake"))

    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert self.matcher._cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert self.matcher._cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert self.matcher._cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        a = [1.0, 2.0]
        b = [0.0, 0.0]
        assert self.matcher._cosine_similarity(a, b) == 0.0


class TestComputeScore:
    """Tests for SkillMatcher._compute_score with mocked embeddings."""

    def setup_method(self):
        self.matcher = SkillMatcher(Config(gemini_api_key="fake"))

    def _mock_embed(self, return_vectors):
        """Patch _embed to return preset vectors.

        The first call from match() returns the topic embedding.
        Subsequent calls return domain embeddings passed via `return_vectors`.
        """
        self.matcher._embed = MagicMock(side_effect=lambda texts: return_vectors[:len(texts)])

    def test_compute_score_domain_only(self):
        """Score = 0.5 * domain_sim when no keywords or intents."""
        topic_emb = [1.0, 0.0, 0.0]
        domain_emb = [1.0, 0.0, 0.0]  # identical → sim = 1.0
        self.matcher._embed = MagicMock(return_value=[domain_emb])

        skill = _make_skill(domains=["test domain"])
        score, domains, keywords = self.matcher._compute_score("test", topic_emb, skill)

        # 0.5 * 1.0 = 0.5
        assert score == pytest.approx(0.5)
        assert "test domain" in domains

    def test_compute_score_keyword_only(self):
        """Score = 0.3 * keyword_ratio when no domains or intents."""
        topic_emb = [1.0, 0.0]
        skill = _make_skill(keywords=["test", "verify", "missing"])
        self.matcher._embed = MagicMock(return_value=[])

        score, _, matched_kw = self.matcher._compute_score("test and verify", topic_emb, skill)

        # 0.3 * (2/3)
        assert score == pytest.approx(0.3 * (2.0 / 3.0))
        assert set(matched_kw) == {"test", "verify"}

    def test_compute_score_intent_only(self):
        """Score = 0.2 * intent_ratio when no domains or keywords."""
        topic_emb = [1.0]
        skill = _make_skill(intent_patterns=[r"test.*code", r"no_match"])
        self.matcher._embed = MagicMock(return_value=[])

        score, _, _ = self.matcher._compute_score("test my code", topic_emb, skill)

        # 0.2 * (1/2)
        assert score == pytest.approx(0.1)


class TestMatch:
    """Tests for SkillMatcher.match with mocked embeddings."""

    def setup_method(self):
        self.matcher = SkillMatcher(Config(gemini_api_key="fake", skill_match_top_k=2))

    def test_match_empty_skills(self):
        assert self.matcher.match("anything", []) == []

    def test_match_filters_by_threshold(self):
        """Skills below the threshold are excluded."""
        # Skill with threshold 0.9 — hard to reach
        high_threshold = _make_skill(name="strict", threshold=0.9)
        # Skill with threshold 0.0 — always matches
        low_threshold = _make_skill(name="easy", keywords=["test"], threshold=0.0)

        # _embed returns a simple vector for topic, and nothing for domains
        self.matcher._embed = MagicMock(return_value=[[0.1]])

        matches = self.matcher.match("test", [high_threshold, low_threshold])

        names = [m.skill.metadata.name for m in matches]
        assert "easy" in names
        assert "strict" not in names

    def test_match_top_k_limit(self):
        """Results capped to top_k."""
        skills = [
            _make_skill(name=f"s{i}", keywords=["test"], threshold=0.0)
            for i in range(5)
        ]
        self.matcher._embed = MagicMock(return_value=[[0.1]])

        matches = self.matcher.match("test", skills)
        assert len(matches) <= 2  # top_k = 2

    def test_match_sorted_by_score(self):
        """Results are returned in descending score order."""
        s1 = _make_skill(name="low", threshold=0.0)
        s2 = _make_skill(name="high", keywords=["exact", "match"], threshold=0.0)

        self.matcher._embed = MagicMock(return_value=[[0.1]])

        matches = self.matcher.match("exact match", [s1, s2])
        if len(matches) >= 2:
            assert matches[0].score >= matches[1].score
