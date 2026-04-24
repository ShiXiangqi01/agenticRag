from fastapi import APIRouter, HTTPException

from src.agent.chat_assistant import ChatAssistant
from src.agent.agentic_rag_system_factory import AgenticRagSystemFactory
from src.data.dtos.agent import AgentModel, AgentResponse, MessageRequest, SessionHandle
from src.data.user_image_store import UserImageStore

router = APIRouter(
    prefix="/agents",
    tags=["agents"],
)

agentic_rag_factory = AgenticRagSystemFactory()
user_image_store = UserImageStore()

@router.post("/send_message", response_model=AgentResponse)
async def send_message(request: MessageRequest):
    try:
        agent, session = agentic_rag_factory.get_or_create_agent(
            model_name=request.model_name,
            session=request.session_id,
        )
        base64_image = None
        if request.user_image_id:
            base64_image = user_image_store.load_user_image(request.user_image_id, base64=True)
        response_text, images_list = agent.handle_user_request(
            user_request=request.message + f" user_image_id: {request.user_image_id}" if request.user_image_id else request.message,
            base64_image=base64_image,
        )

        return AgentResponse(
            message=response_text,
            session=session,
            images_list=images_list,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions", response_model=list[SessionHandle])
async def list_sessions():
    try:
        sessions = agentic_rag_factory.get_all_sessions()
        return sessions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}", response_model=dict)
async def delete_session(session_id: str):
    try:
        success = agentic_rag_factory.delete_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        return {"success": True, "message": f"Session {session_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/available_models", response_model=list[AgentModel])
async def get_available_models():
    try:
        available_models_df = ChatAssistant.list_available_models()
        available_models = available_models_df.to_dict(orient="records")
        return available_models
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
