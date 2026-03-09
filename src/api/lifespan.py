from contextlib import asynccontextmanager
import os

import mlflow
from fastapi import FastAPI
from loguru import logger
from dotenv import load_dotenv
load_dotenv()

# from fundus_murag.agent.chat_assistant_factory import ChatAssistantFactory
# from fundus_murag.agent.fundus_multi_agent_system_factory import FundusMultiAgentSystemFactory
# from fundus_murag.data.user_image_store import UserImageStore
from src.data.vector_db import VectorDB


@asynccontextmanager
async def api_lifespan(app: FastAPI):
    # Startup
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("/agentic")
    mlflow.openai.autolog()
    logger.info("Starting Agentic! Data API")
    # chat_assistant_factory = ChatAssistantFactory()
    # fundus_agent_factory = FundusMultiAgentSystemFactory()
    vdb = VectorDB()
    # user_image_store = UserImageStore()
    yield
    # Shutdown
    vdb.close()
    del vdb

    logger.info("Stopping Agentic! Data API")
