import re
import json
from typing import Any
from enum import Enum
from loguru import logger
from pydantic import BaseModel

from src.config import load_config
from src.data.dtos.session import SessionHandle
from src.agent.chat_assistant import ChatAssistant
from src.agent.chat_assistant_factory import ChatAssistantFactory
from src.agent.tools.tools import(
    get_lookup_tool,
    get_sim_search_tool,
    get_lexical_search_tool,
    get_image_analysis_tool,
)
from src.agent.review_agent import ReviewAgent
from src.data.dtos.agent import AgentResponseImage
from src.data.vector_db import VectorDB
from src.agent.prompts.concierge import (
    CONCIERGE_SYSTEM_INSTRUCTION_TEMPLATE,
    CONCIERGE_ASSISTANTS_LIST_PLACEHOLDER,
    FORWARDING_REQUEST_USER_MESSAGE_TEMPLATE,
    PROCESS_ASSISTANT_RESPONSE_USER_MESSAGE_TEMPLATE,
)
from src.agent.prompts.image_analysis import IMAGE_ANALYSIS_ASSISTANT_SYSTEM_INSTRUCTION
from src.agent.prompts.db_interaction import DB_INTERACTION_ASSISTANT_SYSTEM_INSTRUCTION

class AssistantType(str, Enum):
    CONCIERGE = "concierge"
    DB_LOOKUP = "db_lookup"
    SIM_SEARCH = "sim_search"
    LEX_SEARCH = "lex_search"
    IMG_ANALYSIS = "img_analysis"

class ConciergeAssistant(BaseModel):
    assistant_type: AssistantType
    internal_id: str
    name: str
    description: str

CONCIERGE_ASSISTANTS: dict[AssistantType, ConciergeAssistant] = {
    AssistantType.DB_LOOKUP: ConciergeAssistant(
        assistant_type=AssistantType.DB_LOOKUP,
        internal_id="db_lookup",
        name="Database Lookup Assistant",
        description="This assistant serves as a comprehensive database query interface for AgenticText and AgenticImage data. It can retrieve specific text or image records using their vector IDs and fetch surrounding context chunks. Additionally, it supports exploratory listing operations, allowing you to retrieve all body parts, file names, and character names within specific files from the database.",
    ),
    AssistantType.SIM_SEARCH: ConciergeAssistant(
        assistant_type=AssistantType.SIM_SEARCH,
        internal_id="sim_search",
        name="Similarity Search Assistant",
        description="Semantic similarity search for conceptual or descriptive queries. Use this when the user describes content in natural language, asks for 'similar' items, or has a vague intent (e.g, 'texts about recovery', 'show me something like this'). This should be the default choice for most open-ended questions where no exact ID or specific keyword is provided.",
    ),
    AssistantType.LEX_SEARCH: ConciergeAssistant(
        assistant_type=AssistantType.LEX_SEARCH,
        internal_id="lex_search",
        name="Lexical Search Assistant",
        description="Strict lexical/keyword search for literal string matching. Use this ONLY when the user explicitly requests exact phrase matching, provides technical codes/specific terminology (e.g., 'find images named 'example.jpg'', 'search for file name 'example.txt''). If the request mentions a professional term, lex_search can be used as a supplement to improve recall. Do not use for general natural language questions.",
    ),
    AssistantType.IMG_ANALYSIS: ConciergeAssistant(
        assistant_type=AssistantType.IMG_ANALYSIS,
        internal_id="img_analysis",
        name="Image Analysis Assistant",
        description="Fallback visual analysis assistant for user-uploaded images. Use this when sim_search cannot find similar results or when the user requests OCR/object-level analysis. This assistant expects user_image_id.",
    ),
}

class AgenticRagSystem:
    def __init__(
        self,
        model_name: str | None = None,
    ):
        self._conf = load_config()
        self.model_name = model_name or self._conf.assistant.default_model
        self._assistant_sessions: dict[AssistantType, SessionHandle] = dict()
        self._assistant_factory = ChatAssistantFactory()
        self._review_agent = ReviewAgent(model_name="ollama/qwen3.5:2b")
        self._vdb = VectorDB()
        self._build_assistants()

    def _build_assistants(self) -> None:
        logger.info("Building assistants ...")
        for assistant_type in AssistantType:
            system_instruction = None
            tools = None
            match assistant_type:
                case AssistantType.CONCIERGE:
                    system_instruction = CONCIERGE_SYSTEM_INSTRUCTION_TEMPLATE.replace(
                       CONCIERGE_ASSISTANTS_LIST_PLACEHOLDER, self._generate_concierge_assistants_list()  
                    )
                case AssistantType.DB_LOOKUP:
                    system_instruction = DB_INTERACTION_ASSISTANT_SYSTEM_INSTRUCTION
                    tools = [get_lookup_tool()]
                case AssistantType.SIM_SEARCH:
                    system_instruction = DB_INTERACTION_ASSISTANT_SYSTEM_INSTRUCTION 
                    tools = [get_sim_search_tool()]
                case AssistantType.LEX_SEARCH:
                    system_instruction = DB_INTERACTION_ASSISTANT_SYSTEM_INSTRUCTION 
                    tools = [get_lexical_search_tool()]
                case AssistantType.IMG_ANALYSIS:
                    system_instruction = IMAGE_ANALYSIS_ASSISTANT_SYSTEM_INSTRUCTION 
                    tools = [get_image_analysis_tool()]
                case _:
                    raise ValueError(f"Unsupported assistant type: {assistant_type}")
                
            _, seesion = self._assistant_factory.get_or_create_assistant(
                assistant_name= assistant_type.value.replace("_", " ").upper(),
                model_name=self.model_name,
                system_instruction=system_instruction,
                available_tools=tools,
                session=None,
            )
            self._assistant_sessions[assistant_type] = seesion
            logger.info(f"{assistant_type.upper()} Assistant built successfully.")

    def _generate_concierge_assistants_list(self) -> str:
        assistants_list = """
# Your Assistants

You have the following assistants at your disposal:
"""
        for assistant in CONCIERGE_ASSISTANTS.values():
            assistants_list += f"""
**{assistant.name}**
   name: `{assistant.internal_id}`
   description: {assistant.description}
"""
        return assistants_list    
    
    def _get_assistant(self, assistant_type: AssistantType) -> ChatAssistant:
        if assistant_type not in self._assistant_sessions:
            raise KeyError(f"{assistant_type.upper()} Assistant not found.")
        session = self._assistant_sessions[assistant_type]
        assistant, _ = self._assistant_factory.get_or_create_assistant(
            session = session.session_id,
        )

        return assistant

    def _parse_forwarding_request(self, concierge_response: str) -> dict[str, str] | None:
        resp = concierge_response.encode().decode("unicode_escape").replace("\n", "")

        mandatory_keys = ["assistant", "user_request", "context"]

        # Regex to check if the response contains JSON with or without markdown fences
        json_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```|(\{.*?\})")
        match = json_pattern.search(resp)
        if match:
            json_str = match.group(1) if match.group(1) else match.group(2)
            try:
                json_obj = json.loads(json_str)
                if all(key in json_obj for key in mandatory_keys):
                    return json_obj
                else:
                    msg = f"Malformed Forwarding Request: Missing mandatory keys. {json_obj}"
                    logger.error(msg)
                    raise ValueError(msg)
            except json.JSONDecodeError:
                logger.debug("Found JSON-like content, but failed to parse it")
            
            try:
                json_start = resp.find("{")
                json_end = resp.rfind("}")
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    json_str = resp[json_start : json_end + 1]
                    json_obj = json.loads(json_str)
                    if all(key in json_obj for key in mandatory_keys):
                        return json_obj
                    else:
                        msg = f"Malformed Forwarding Request: Missing mandatory keys. {json_obj}"
                        logger.error(msg)
                        raise ValueError(msg)
            except (json.JSONDecodeError, ValueError, KeyError):
                logger.debug("Failed to extract JSON with the fallback method")

            return None
 
    def _forward_user_request(
        self, 
        forwarding_request: dict[str, str],
        user_request: str,
        base64_image: str | None = None,
    ) -> str:
        assistant_name = forwarding_request["assistant"]
        user_request = forwarding_request["user_request"]
        context = forwarding_request["context"]
        user_image_id = forwarding_request.get("user_image_id", "")
        vector_id = forwarding_request.get("vector_id", "")

        try:
            assistant_type = AssistantType(assistant_name)
        except ValueError:
            raise ValueError(f"Unsupported Assistant name: {assistant_name}")

        assistant = self._get_assistant(assistant_type)
        
        message = FORWARDING_REQUEST_USER_MESSAGE_TEMPLATE.format(
            USER_REQUEST=user_request,
            CONTEXT=context,
            USER_IMAGE_ID=user_image_id,
            VECTOR_ID=vector_id
        )
        assistant_response = assistant.send_user_message(text_message=message, base64_image=base64_image)
        return assistant_response
    
    
    def _process_assistant_response(
        self,
        assistant_response: str,
        original_user_request: str,
        forwarded_request: dict[str, str],
    ) -> str:
        concierge_assistant = self._get_assistant(AssistantType.CONCIERGE)

        assistant_response_message = PROCESS_ASSISTANT_RESPONSE_USER_MESSAGE_TEMPLATE.format(
            ORIGINAL_USER_REQUEST=original_user_request,
            FORWARDED_REQUEST=json.dumps(forwarded_request, indent=2),
            ASSISTANT_NAME=forwarded_request["assistant"],
            ASSISTANT_RESPONSE=assistant_response,
        )
        
        concierge_response = concierge_assistant.send_user_message(text_message=assistant_response_message)

        return concierge_response

    def _is_review_pass(self, review_response: dict[str, Any]) -> bool:
        intent_match = str(review_response.get("intent_match", "fail")).strip().lower()
        action = str(review_response.get("action", "rewrite")).strip().lower()
        notes = str(review_response.get("notes", "fail")).strip().lower()

        return (
            intent_match == "pass"
            and action == "accept"
        )

    def _build_review_feedback_message(
        self,
        original_user_request: str,
        review_response: dict[str, Any],
    ) -> str:
        return (
            "Please rewrite your last final answer according to the review feedback below.\n"
            "Return only the final user-facing answer in the same language as the user request.\n\n"
            f"Original User Request:\n{original_user_request}\n\n"
            "Review Feedback JSON:\n"
            f"{json.dumps(review_response, ensure_ascii=False, indent=2)}"
        )

    def _resolve_forwarding_chain(
        self,
        concierge_response: str,
        user_request: str,
        base64_image: str | None = None,
    ) -> str:
        forwarding_request = self._parse_forwarding_request(concierge_response)
        while forwarding_request is not None:
            assistant_response = self._forward_user_request(
                forwarding_request,
                user_request,
                base64_image,
            )

            concierge_response = self._process_assistant_response(
                assistant_response=assistant_response,
                original_user_request=user_request,
                forwarded_request=forwarding_request,
            )
            forwarding_request = self._parse_forwarding_request(concierge_response)

        return concierge_response

    def _process_review_response(self, review_response: dict[str, Any], draft_response: str) -> tuple[str, list[AgentResponseImage]]:
        is_pass = self._is_review_pass(review_response)

        if is_pass:
            response_to_render = str(review_response.get("final_response", "")).strip() or draft_response
        else:
            intent_match = str(review_response.get("intent_match", "fail")).strip().lower()
            action = str(review_response.get("action", "rewrite")).strip().lower()
            notes = str(review_response.get("notes", "fail")).strip().lower()
            logger.info(
                "Review not fully passed; fallback to draft response. "
                f"intent_match={intent_match}, action={action}, notes={notes}"
            )
            # response_to_render = draft_response

        image_tag_pattern = re.compile(
            r"<agentic_image\s+[^>]*?vector_id\s*=\s*['\"]([^'\"]+)['\"][^>]*/>",
            flags=re.IGNORECASE,
        )

        vector_ids: list[str] = []
        for match in image_tag_pattern.finditer(response_to_render):
            vector_id = match.group(1).strip()
            if vector_id and vector_id not in vector_ids:
                vector_ids.append(vector_id)

        images_list: list[AgentResponseImage] = []
        for vector_id in vector_ids:
            try:
                image_record = self._vdb.get_base_image_by_vector_id(vector_id)
                images_list.append(
                    AgentResponseImage(
                        vector_id=str(image_record.vector_id),
                        image_name=image_record.image_name,
                        image_path=image_record.image_path,
                        base64_image=image_record.base64_image,
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to resolve image tag vector_id={vector_id}: {e}")

        response_text = image_tag_pattern.sub("", response_to_render)
        response_text = re.sub(r"\n{3,}", "\n\n", response_text).strip()

        return response_text, images_list

      

    def handle_user_request(
        self,
        user_request: str,
        base64_image: str | None = None,
    ) -> tuple[str, list[AgentResponseImage]]:
        MAX_REVIEW_RETRY = 1

        # 1. the user request is sent to the concierge
        concierge_assistant = self._get_assistant(AssistantType.CONCIERGE)
        concierge_response = concierge_assistant.send_user_message(text_message=user_request, base64_image=base64_image)

        # 2. run forwarding chain until concierge returns a final draft
        concierge_response = self._resolve_forwarding_chain(
            concierge_response=concierge_response,
            user_request=user_request,
            base64_image=base64_image,
        )

        # 3. review loop: if review is not pass, send feedback to concierge for rewrite
        for retry_idx in range(MAX_REVIEW_RETRY + 1):
            review_response = self._review_agent.review_response(
                original_user_request=user_request,
                draft_final_response=concierge_response,
            )

            if self._is_review_pass(review_response):
                return self._process_review_response(review_response, concierge_response)

            if retry_idx >= MAX_REVIEW_RETRY:
                logger.warning("Review did not pass after retries; returning best effort draft response.")
                return self._process_review_response(review_response, concierge_response)

            feedback_message = self._build_review_feedback_message(
                original_user_request=user_request,
                review_response=review_response,
            )
            concierge_response = concierge_assistant.send_user_message(
                text_message=feedback_message,
                base64_image=base64_image,
            )
            concierge_response = self._resolve_forwarding_chain(
                concierge_response=concierge_response,
                user_request=user_request,
                base64_image=base64_image,
            )

        return self._process_review_response(
            review_response={
                "intent_match": "fail",
                "action": "rewrite",
                "notes": "fail",
                "final_response": concierge_response,
            },
            draft_response=concierge_response,
        )

        