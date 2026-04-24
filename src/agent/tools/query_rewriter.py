
import re
from collections import OrderedDict
from enum import Enum
from loguru import logger
from src.singleton_meta import SingletonMeta
from src.agent.chat_assistant_factory import ChatAssistantFactory
from src.agent.chat_assistant import ChatAssistant
from src.agent.prompts.query_rewriting import (
    QUERY_REWRITER_TEXT_IMAGE_SYSTEM_INSTRUCTION,
    QUERY_REWRITER_TEXT_TEXT_SYSTEM_INSTRUCTION,
)


class QueryRewritingTask(str, Enum):
    T2T = "t2t"
    T2I = "t2i"


QUERY_REWRITER_GENERATION_CONFIG = {
    "n": 1,
    "temperature": 0.1,
    "max_completion_tokens": 128,
}

MAX_REWRITE_CACHE_SIZE = 256
FAST_PATH_MAX_CHARS = 160
FAST_PATH_MAX_TOKENS = 16
QUERY_REWRITER_TIMEOUT_SECONDS = 60.0


COMPLEX_QUERY_HINTS = {
    "how", "why", "what", "which", "when", "where",
    "compare", "difference", "versus", "vs",
    "step", "steps", "plan", "routine",
    "contraindication", "precaution", "warning", "risk",
    "symptom", "cause", "diagnosis", "treatment",
    "how to", "explain",
    "怎么", "为什么", "区别", "步骤", "注意", "禁忌",
}

class QueryRewriter(metaclass=SingletonMeta):
    def __init__(self):
        self._factory = ChatAssistantFactory()
        self._rewrite_cache: OrderedDict[tuple[str, str], str] = OrderedDict()

    @staticmethod
    def _normalize_query(user_query: str) -> str:
        return re.sub(r"\s+", " ", user_query).strip()

    def _cache_get(self, task: QueryRewritingTask, user_query: str) -> str | None:
        key = (task.value, self._normalize_query(user_query).lower())
        cached = self._rewrite_cache.get(key)
        if cached is not None:
            # Refresh insertion order for LRU behavior.
            self._rewrite_cache.move_to_end(key)
        return cached

    def _cache_set(self, task: QueryRewritingTask, user_query: str, rewritten_query: str) -> None:
        key = (task.value, self._normalize_query(user_query).lower())
        self._rewrite_cache[key] = rewritten_query
        self._rewrite_cache.move_to_end(key)
        if len(self._rewrite_cache) > MAX_REWRITE_CACHE_SIZE:
            self._rewrite_cache.popitem(last=False)

    def _is_fast_path_query(self, user_query: str) -> bool:
        normalized = self._normalize_query(user_query)
        if not normalized:
            return True

        lowered = normalized.lower()
        if any(hint in lowered for hint in COMPLEX_QUERY_HINTS):
            return False

        tokens = normalized.split(" ")
        return len(normalized) <= FAST_PATH_MAX_CHARS and len(tokens) <= FAST_PATH_MAX_TOKENS

    def _rewrite(self, task: QueryRewritingTask, user_query: str) -> str:
        normalized = self._normalize_query(user_query)
        if self._is_fast_path_query(normalized):
            logger.debug(f"[QueryRewriter] Fast-path skip for short/simple query: {normalized}")
            return normalized

        cached = self._cache_get(task, normalized)
        if cached is not None:
            logger.debug("[QueryRewriter] Cache hit for rewritten query")
            return cached

        try:
            assistant = self._get_query_rewriter_assistant(task)
            rewritten_query = self._normalize_query(assistant.send_user_message(normalized))
            if rewritten_query:
                self._cache_set(task, normalized, rewritten_query)
                return rewritten_query
        except Exception as e:
            logger.error(f"Error rewriting user query for task={task.value}: {e}")

        return normalized

    def _get_query_rewriter_assistant(self, task: QueryRewritingTask)-> ChatAssistant:
        match task:
            case QueryRewritingTask.T2T:
                system_instruction = QUERY_REWRITER_TEXT_TEXT_SYSTEM_INSTRUCTION
            case QueryRewritingTask.T2I:
                system_instruction = QUERY_REWRITER_TEXT_IMAGE_SYSTEM_INSTRUCTION
            case _:
                raise ValueError(f"Unsupported query rewriting task: {task}")

        # Use a fresh assistant session per rewrite call to prevent chat history
        # growth from increasing prompt size and latency over time.
        assistant, _ = self._factory.get_or_create_assistant(
            assistant_name="Query Rewriter",
            model_name = "ollama/qwen3.5:2b",
            system_instruction = system_instruction,
            available_tools = None,
            generation_config = QUERY_REWRITER_GENERATION_CONFIG,
            request_timeout = QUERY_REWRITER_TIMEOUT_SECONDS,
            session = None,
        )

        return assistant
    
    def rewrite_user_query_for_cross_modal_text_to_image_search(self, user_query: str) -> str:
        """
        Rewrites a user query to enhance the results of a cross-modal text-image search.

        Args:
            user_query: The user query to rewrite.

        Returns:
            The rewritten user query optimized for cross-modal text-image search.
        """
        rewritten_query = self._rewrite(QueryRewritingTask.T2I, user_query)

        logger.debug(f"User Query: {user_query}")
        logger.debug(f"Rewritten Query: {rewritten_query}")

        return rewritten_query

    def rewrite_user_query_for_text_to_text_search(self, user_query: str) -> str: 
        """
        """
        rewritten_query = self._rewrite(QueryRewritingTask.T2T, user_query)

        logger.debug(f"User Query: {user_query}")
        logger.debug(f"Rewritten Query: {rewritten_query}")

        return rewritten_query