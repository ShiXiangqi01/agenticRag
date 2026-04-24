

import time
from typing import TypeVar, Generic
import uuid
from loguru import logger

from src.data.dtos.session import SessionHandle


MAX_SESSION_AGE = 60 * 60  # 1 hour
MAX_SESSIONS = 100

T = TypeVar("T") #TypeVar 是 Python typing 模块里的“类型变量”，用来写泛型。简单说：它表示“这里的类型先不固定，等使用时再确定”。

class SessionManager(Generic[T]):
    def __init__(self, clazz: type[T]):
        self.__session_objects: dict[str, T] = {}
        self.__sessions:dict[str, SessionHandle] = {}
        self.__old_sessions: dict[str, SessionHandle] = {}
        self._concrete_class_name = clazz.__name__
        logger.info(f"Initialized SessionManager for {self._concrete_class_name}")

    def __get_session(self, session_id: str) -> SessionHandle:
        if session_id in self.__sessions:
            session = self.__sessions[session_id]
            logger.debug(f"Session {session_id} found for {self._concrete_class_name}")
            session.updated = int(time.time())
            session.expires = session.updated + MAX_SESSION_AGE
            self.__sessions[session_id] = session
            return session
        elif session_id in self.__old_sessions:
            logger.error(f"Session {session_id} for {self._concrete_class_name} has expired")
            raise ValueError(f"Session {session_id} has expired.")
        else:
            logger.error(f"Session {session_id} for {self._concrete_class_name} not found")
            raise ValueError(f"Session {session_id} not found.")
        
    def get_all_sessions(self) -> list[SessionHandle]:
        sessions = []
        for session in self.__sessions.values():
            sessions.append(session.model_copy())
        return sessions
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID. Returns True if session was deleted, False if not found."""
        return self.__delete_session(session_id)
        
    def __create_session(
        self,
        cls: type[T],
        *args,
        **kwargs,
    ) -> tuple[T, SessionHandle]:
        logger.info(f"Creating new session for {self._concrete_class_name}")
        obj = cls(*args, **kwargs)
        session_id = str(uuid.uuid4())
        now = int(time.time())
        session = SessionHandle(
            session_id=session_id,
            created=now,
            updated=now,
            expires=now + MAX_SESSION_AGE,
        )
        self.__sessions[session_id] = session
        self.__session_objects[session_id] = obj
        logger.debug(f"Session {session_id} created for {self._concrete_class_name}")
        return obj, session

    def __delete_session(self, session_id: str) -> bool:
        if session_id in self.__sessions:
            session = self.__sessions[session_id]
            self.__old_sessions[session_id] = session.model_copy() #what is model_copy ? copy the model to old session dict before deleting it from current session dict, so that we can keep track of expired sessions for better error handling and debugging
            del self.__sessions[session_id]
            del self.__session_objects[session_id]
            logger.debug(f"Session {session_id} deleted for {self._concrete_class_name}")
            return True
        return False

    def __cleanup_up_sessions(self) -> None:
        current_time = int(time.time())
        session_ids = list(self.__sessions.keys())
        for session_id in session_ids:
            session = self.__sessions[session_id]
            if current_time > session.expires:
                self.__delete_session(session_id)

    def get_or_create_session(
            self,
            cls: type[T],
            session: str | SessionHandle | None = None,
            *args,
            **kwargs,
    ) -> tuple[T, SessionHandle]:
        self.__cleanup_up_sessions() 
        if session is not None and session.session_id != "":
            session = self.__get_session(session.session_id) 
            obj = self.__session_objects[session.session_id] #what is session object ? it more details about the session, like the chat history, the model used, the system instruction, and the available tools for this session, etc.
        else:
            obj, session = self.__create_session(cls, *args, **kwargs)
        return obj, session
        

