"""LightRAG knowledge graph builder."""

import asyncio
import os
from pathlib import Path

from kbskills.config import Config
from kbskills.utils.retry import retry_api_call, KnowledgeBaseError


_rag_instance = None


def _run_async(coro):
    """Run an async coroutine from sync code."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already in an async context — create a new thread to run it
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


def get_rag_instance(config: Config):
    """Get or create a LightRAG instance with Gemini backend."""
    global _rag_instance
    if _rag_instance is not None:
        return _rag_instance

    import numpy as np
    from google import genai
    from lightrag import LightRAG
    from lightrag.llm.gemini import gemini_model_complete
    from lightrag.utils import EmbeddingFunc

    # Set API key in environment for LightRAG's Gemini integration
    os.environ["GEMINI_API_KEY"] = config.gemini_api_key

    embedding_model = f"models/{config.embedding_model}"
    _genai_client = genai.Client(api_key=config.gemini_api_key)

    async def gemini_embedding(texts: list[str]) -> np.ndarray:
        """Custom embedding function that returns (n, dim) ndarray.

        Note: Retries are handled at a higher level by the caller.
        """
        try:
            result = _genai_client.models.embed_content(
                model=embedding_model,
                contents=texts,
            )
            return np.array([e.values for e in result.embeddings])
        except Exception as e:
            raise KnowledgeBaseError(f"Embedding failed during graph operation: {e}") from e

    working_dir = str(Path(config.data_dir) / "graph")
    Path(working_dir).mkdir(parents=True, exist_ok=True)

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=gemini_model_complete,
        llm_model_name=config.llm_model,
        embedding_func=EmbeddingFunc(
            embedding_dim=3072,
            max_token_size=8192,
            func=gemini_embedding,
        ),
    )

    # LightRAG v1.4.9+ requires async storage initialization
    _run_async(rag.initialize_storages())

    _rag_instance = rag
    return _rag_instance


def reset_rag_instance():
    """Reset the cached RAG instance (useful for testing)."""
    global _rag_instance
    _rag_instance = None


@retry_api_call(operation_name="KnowledgeBase", max_retries=3, min_wait=2, max_wait=20)
def query_knowledge(config: Config, query: str, mode: str = "hybrid") -> str:
    """Query the knowledge graph.

    Retries up to 3 times with exponential backoff on query failures.

    Args:
        config: Application configuration.
        query: The search query.
        mode: Search mode - one of "naive", "local", "global", "hybrid".

    Returns:
        Retrieved context as text.

    Raises:
        KnowledgeBaseError: If all retries are exhausted.
    """
    try:
        from lightrag import QueryParam

        rag = get_rag_instance(config)
        result = rag.query(query, param=QueryParam(mode=mode))
        return result
    except KnowledgeBaseError:
        raise
    except Exception as e:
        raise KnowledgeBaseError(f"Knowledge base query failed: {e}") from e
