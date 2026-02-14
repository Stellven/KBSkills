"""Unit tests for kbskills.utils.retry module."""

import pytest

from kbskills.utils.retry import (
    KBSkillsError,
    LLMError,
    EmbeddingError,
    KnowledgeBaseError,
    IngestionError,
    retry_llm_call,
    retry_embedding_call,
    retry_api_call,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MIN_WAIT,
    DEFAULT_MAX_WAIT,
    DEFAULT_MULTIPLIER,
)


class TestExceptionHierarchy:

    def test_llm_error_is_kbskills_error(self):
        assert issubclass(LLMError, KBSkillsError)

    def test_embedding_error_is_kbskills_error(self):
        assert issubclass(EmbeddingError, KBSkillsError)

    def test_knowledge_base_error_is_kbskills_error(self):
        assert issubclass(KnowledgeBaseError, KBSkillsError)

    def test_ingestion_error_is_kbskills_error(self):
        assert issubclass(IngestionError, KBSkillsError)

    def test_kbskills_error_is_exception(self):
        assert issubclass(KBSkillsError, Exception)

    def test_exception_message(self):
        err = LLMError("test message")
        assert str(err) == "test message"


class TestRetryConstants:

    def test_default_max_retries(self):
        assert DEFAULT_MAX_RETRIES == 3

    def test_default_min_wait(self):
        assert DEFAULT_MIN_WAIT == 2

    def test_default_max_wait(self):
        assert DEFAULT_MAX_WAIT == 30

    def test_default_multiplier(self):
        assert DEFAULT_MULTIPLIER == 2


class TestRetryLLMCall:

    def test_success_no_retry(self):
        @retry_llm_call(max_retries=3, min_wait=0, max_wait=0)
        def succeed():
            return "ok"

        assert succeed() == "ok"

    def test_retries_then_succeeds(self):
        call_count = 0

        @retry_llm_call(max_retries=3, min_wait=0, max_wait=0)
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("transient error")
            return "recovered"

        assert flaky() == "recovered"
        assert call_count == 3

    def test_exhausted_retries_raises(self):
        @retry_llm_call(max_retries=2, min_wait=0, max_wait=0)
        def always_fail():
            raise ValueError("permanent error")

        with pytest.raises(ValueError, match="permanent error"):
            always_fail()


class TestRetryEmbeddingCall:

    def test_success(self):
        @retry_embedding_call(max_retries=2, min_wait=0, max_wait=0)
        def embed():
            return [0.1, 0.2]

        assert embed() == [0.1, 0.2]

    def test_retry_on_failure(self):
        attempt = 0

        @retry_embedding_call(max_retries=3, min_wait=0, max_wait=0)
        def flaky_embed():
            nonlocal attempt
            attempt += 1
            if attempt < 2:
                raise RuntimeError("embedding timeout")
            return [0.5]

        assert flaky_embed() == [0.5]
        assert attempt == 2


class TestRetryApiCall:

    def test_success(self):
        @retry_api_call(operation_name="Test", max_retries=2, min_wait=0, max_wait=0)
        def api():
            return {"status": "ok"}

        assert api() == {"status": "ok"}

    def test_exhausted_raises(self):
        @retry_api_call(operation_name="Test", max_retries=2, min_wait=0, max_wait=0)
        def broken_api():
            raise ConnectionError("network down")

        with pytest.raises(ConnectionError, match="network down"):
            broken_api()
