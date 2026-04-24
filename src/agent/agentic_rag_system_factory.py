from src.singleton_meta import SingletonMeta
from src.agent.session_manager import SessionManager
from src.data.dtos.session import SessionHandle
from src.agent.agentic_rag_system import AgenticRagSystem


class AgenticRagSystemFactory(metaclass=SingletonMeta):
    def __init__(self):
        self.__session_manager = SessionManager[AgenticRagSystem](AgenticRagSystem)
    
    def get_or_create_agent(
            self,
            model_name: str | None,
            session: str | SessionHandle | None = None,
    ) -> tuple[AgenticRagSystem, SessionHandle]:
        if isinstance(session, str):
            session = SessionHandle(
                session_id=session,
                created=-1,
                updated=-1,
                expires=-1,
            )
        assistant, session = self.__session_manager.get_or_create_session(
            AgenticRagSystem,
            model_name=model_name,
            session=session,
        )
        return assistant, session   
    
    def get_all_sessions(self) -> list[SessionHandle]:
        return self.__session_manager.get_all_sessions()
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID."""
        return self.__session_manager.delete_session(session_id)