from pydantic import BaseModel, Field
from src.data.dtos.session import SessionHandle


class AgentResponseImage(BaseModel):
    vector_id: str
    image_name: str
    image_path: str
    base64_image: bytes

class AgentModel(BaseModel):
    name: str
    display_name: str

class AgentResponse(BaseModel):
    message: str
    session: SessionHandle
    images_list: list[AgentResponseImage] = Field(default_factory=list)

class ChatMessage(BaseModel):
    role: str
    content: str


class MessageRequest(BaseModel):

    message: str = Field(..., description="The text message to send to the  AgenticRag Agent.")
    user_image_id: str | None = Field(
        None,
        description="The ID of the user-provided image, which was uploaded through the API before. If None, no image is provided.",
    )
    model_name: str | None = Field(
        "ollama/qwen3.5:27b ",
        description="The name of the model to use for the AgenticRag Agent. The model must be available in the system. If a session ID is provided, this field is ignored.",
    )
    session_id: str | None = Field(
        None,
        description="The session ID to use for the AgenticRag Agent. If None, a new session is created.",
    )