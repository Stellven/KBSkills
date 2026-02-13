"""Retry and resilience utilities for external API calls."""

import logging
from functools import wraps

from rich.console import Console

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

console = Console()
logger = logging.getLogger(__name__)

# ── Custom Exceptions ────────────────────────────────────────────────────────

class KBSkillsError(Exception):
    """Base exception for KBSkills."""
    pass


class LLMError(KBSkillsError):
    """Error during LLM API calls."""
    pass


class EmbeddingError(KBSkillsError):
    """Error during embedding API calls."""
    pass


class KnowledgeBaseError(KBSkillsError):
    """Error interacting with the knowledge base (LightRAG)."""
    pass


class IngestionError(KBSkillsError):
    """Error during data ingestion."""
    pass


# ── Retry Constants ──────────────────────────────────────────────────────────

DEFAULT_MAX_RETRIES = 3
DEFAULT_MIN_WAIT = 2        # seconds
DEFAULT_MAX_WAIT = 30       # seconds
DEFAULT_MULTIPLIER = 2      # exponential backoff multiplier


# ── Retry Decorators ─────────────────────────────────────────────────────────

def retry_llm_call(
    max_retries: int = DEFAULT_MAX_RETRIES,
    min_wait: int = DEFAULT_MIN_WAIT,
    max_wait: int = DEFAULT_MAX_WAIT,
):
    """Retry decorator for LLM (Gemini generate_content) calls.

    Retries on general exceptions (API errors, timeouts, rate limits),
    with exponential backoff. Re-raises the original exception after exhausting retries.
    """
    def decorator(func):
        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=DEFAULT_MULTIPLIER, min=min_wait, max=max_wait),
            retry=retry_if_exception_type((Exception,)),
            before_sleep=_log_retry("LLM"),
            reraise=True,
        )
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def retry_embedding_call(
    max_retries: int = DEFAULT_MAX_RETRIES,
    min_wait: int = DEFAULT_MIN_WAIT,
    max_wait: int = DEFAULT_MAX_WAIT,
):
    """Retry decorator for embedding API calls."""
    def decorator(func):
        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=DEFAULT_MULTIPLIER, min=min_wait, max=max_wait),
            retry=retry_if_exception_type((Exception,)),
            before_sleep=_log_retry("Embedding"),
            reraise=True,
        )
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def retry_api_call(
    operation_name: str = "API",
    max_retries: int = DEFAULT_MAX_RETRIES,
    min_wait: int = DEFAULT_MIN_WAIT,
    max_wait: int = DEFAULT_MAX_WAIT,
):
    """General-purpose retry decorator for external API calls."""
    def decorator(func):
        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=DEFAULT_MULTIPLIER, min=min_wait, max=max_wait),
            retry=retry_if_exception_type((Exception,)),
            before_sleep=_log_retry(operation_name),
            reraise=True,
        )
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ── Logging Helper ───────────────────────────────────────────────────────────

def _log_retry(operation: str):
    """Return a before_sleep callback that logs retry info to both Rich console and logger."""
    def callback(retry_state):
        attempt = retry_state.attempt_number
        wait = retry_state.next_action.sleep if retry_state.next_action else 0
        exception = retry_state.outcome.exception() if retry_state.outcome else None

        exc_type = type(exception).__name__ if exception else "Unknown"
        exc_msg = str(exception)[:200] if exception else ""

        msg = (
            f"[yellow]{operation} call failed (attempt {attempt}): "
            f"[{exc_type}] {exc_msg}. Retrying in {wait:.1f}s...[/yellow]"
        )
        console.print(msg)
        logger.warning(
            "%s call failed (attempt %d): [%s] %s. Retrying in %.1fs...",
            operation, attempt, exc_type, exc_msg, wait,
        )
    return callback
