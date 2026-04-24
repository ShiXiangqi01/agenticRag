from contextlib import asynccontextmanager
import os

import mlflow
from fastapi import FastAPI
from loguru import logger
from dotenv import load_dotenv
load_dotenv()

from src.data.user_image_store import UserImageStore
from src.data.vector_db import VectorDB
from src.agent.chat_assistant_factory import ChatAssistantFactory
from src.agent.agentic_rag_system_factory import AgenticRagSystemFactory


@asynccontextmanager
async def api_lifespan(app: FastAPI):
    # Startup
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("/agentic")
    mlflow.openai.autolog()
    logger.info("Starting Agentic! Data API")
    chat_assistant_factory = ChatAssistantFactory()
    agentic_rag_system_factory= AgenticRagSystemFactory()
    vdb = VectorDB()
    user_image_store = UserImageStore()
    yield
    # Shutdown
    vdb.close()
    del vdb
    del chat_assistant_factory
    del agentic_rag_system_factory
    del user_image_store

    logger.info("Stopping Agentic! Data API")
