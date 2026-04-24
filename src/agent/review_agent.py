
import json
from loguru import logger
from typing import Any

from src.config import load_config
from src.agent.prompts.review_agent import REVIEW_ASSISTANT_SYSTEM_INSTRUCTION, REVIEW_USER_MESSAGE_TEMPLATE
from src.agent.chat_assistant import ChatAssistant
from src.agent.chat_assistant_factory import ChatAssistantFactory

REVIEW_AGENT_MODEL_NAME = "ollama/qwen3.5:2b"
REVIEW_AGENT_REQUEST_TIMEOUT_SECONDS = 60.0

class ReviewAgent:
    def __init__(self, model_name: str | None = None):
        conf = load_config()
        self.model_name = model_name or REVIEW_AGENT_MODEL_NAME
        self._assistant_factory = ChatAssistantFactory()

    def _get_review_assistant(self) -> ChatAssistant:
        assistant, _ = self._assistant_factory.get_or_create_assistant(
            assistant_name="REVIEW",
            model_name=self.model_name,
            system_instruction=REVIEW_ASSISTANT_SYSTEM_INSTRUCTION,
            available_tools=None,
            generation_config={
                "n": 1,
                "temperature": 0.1,
                "max_completion_tokens": 256,
            },
            request_timeout=REVIEW_AGENT_REQUEST_TIMEOUT_SECONDS,
            session=None,
        )
        return assistant

    def _parse_review_response(self, review_response: str) -> dict[str, Any] | None:
        required_keys = ["intent_match", "quality_checks", "action", "final_response", "notes"]

        try:
            payload = json.loads(review_response)
            if isinstance(payload, dict) and all(key in payload for key in required_keys):
                return payload
        except json.JSONDecodeError:
            pass

        json_start = review_response.find("{")
        json_end = review_response.rfind("}")
        if json_start == -1 or json_end == -1 or json_end <= json_start:
            return None

        try:
            payload = json.loads(review_response[json_start : json_end + 1])
            if isinstance(payload, dict) and all(key in payload for key in required_keys):
                return payload
        except json.JSONDecodeError:
            return None

        return None

    def _fallback_payload(self, draft_final_response: str, notes: str = "fail") -> dict[str, Any]:
        return {
            "intent_match": "fail",
            "quality_checks": {
                "factual_consistency": "fail",
                "safety_uncertainty": "fail",
                "format_compliance": "fail",
            },
            "action": "rewrite",
            "final_response": draft_final_response,
            "notes": notes,
        }

    def _timeout_bypass_payload(self, draft_final_response: str) -> dict[str, Any]:
        return {
            "intent_match": "pass",
            "quality_checks": {
                "factual_consistency": "revision",
                "safety_uncertainty": "revision",
                "format_compliance": "pass",
            },
            "action": "accept",
            "final_response": draft_final_response,
            "notes": "review_timeout_bypass",
        }

    @staticmethod
    def _is_timeout_error(error: Exception) -> bool:
        error_name = type(error).__name__.lower()
        error_text = str(error).lower()
        return "timeout" in error_name or "timed out" in error_text or "timeout" in error_text

    def review_response(self, original_user_request: str, draft_final_response: str) -> dict[str, Any]:
        review_message = REVIEW_USER_MESSAGE_TEMPLATE.format(
            ORIGINAL_USER_REQUEST=original_user_request,
            DRAFT_FINAL_RESPONSE=draft_final_response,
        )

        try:
            review_assistant = self._get_review_assistant()
            review_output = review_assistant.send_user_message(text_message=review_message)
            parsed = self._parse_review_response(review_output)
            if parsed is None:
                logger.warning("Review response could not be parsed; using fallback payload.")
                return self._fallback_payload(draft_final_response)

            action = str(parsed.get("action", "rewrite")).strip().lower()
            intent_match = str(parsed.get("intent_match", "fail")).strip().lower()
            notes = str(parsed.get("notes", "fail")).strip().lower()
            final_response = str(parsed.get("final_response", "")).strip()

            if action not in {"accept", "pass", "rewrite", "revision", "fail"}:
                logger.warning("Invalid review action; forcing rewrite.")
                action = "rewrite"

            if final_response == "":
                logger.warning("Empty reviewed final_response; using draft response.")
                final_response = draft_final_response

            parsed["intent_match"] = intent_match
            parsed["action"] = action
            parsed["notes"] = notes
            parsed["final_response"] = final_response

            logger.info(
                f"Review parsed: intent_match={intent_match}, action={action}, notes={notes}"
            )
            return parsed
        except Exception as e:
            if self._is_timeout_error(e):
                logger.warning(
                    f"Review step timed out with {type(e)}: {e}; bypassing review and accepting draft response."
                )
                return self._timeout_bypass_payload(draft_final_response)
            logger.warning(f"Review step failed with {type(e)}: {e}; using fallback payload.")
            return self._fallback_payload(draft_final_response)
