from typing import Any

from src.singleton_meta import SingletonMeta
from src.agent.chat_assistant import ChatAssistant
from src.agent.tools.tools import Tool
from src.data.dtos.session import SessionHandle
from src.agent.session_manager import SessionManager


class ChatAssistantFactory(metaclass=SingletonMeta):
    def __init__(self):
        self.__session_manager = SessionManager[ChatAssistant](ChatAssistant)

    def get_or_create_assistant(
            self,
            assistant_name: str | None = None,
            model_name: str | None = None,
            system_instruction: str | None = None,
            available_tools: list[Tool] | None = None,
            generation_config: dict[str, Any] | None = None,
            request_timeout: float | None = None,
            session: str | SessionHandle | None = None,
    )-> tuple[ChatAssistant, SessionHandle]:
        if isinstance(session, str):
            session = SessionHandle(
                session_id=session,
                created = -1,
                updated = -1,
                expires = -1,
            )
        assistant, session = self.__session_manager.get_or_create_session(
            ChatAssistant,
            assistant_name=assistant_name,
            model_name=model_name,
            system_instruction=system_instruction,
            available_tools=available_tools,
            generation_config=generation_config,
            request_timeout=request_timeout,
            session=session,
        )
        return assistant, session     
    
    def get_all_sessions(self) -> list[SessionHandle]:
        return self.__session_manager.get_all_sessions()
