from pydantic import BaseModel, Field
import uuid

class PostgreFile(BaseModel):
    file_name_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    file_name: str

class PostgreImage(BaseModel):
    postgre_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    vector_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    image_name: str
    page_number: int
    image_index: int
    image_path: str
    file_name: str
    description: str
    body_part: str
    base64_image: bytes
class PostgreText(BaseModel):
    doc_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    vector_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    character_path: str
    file_name: str
    body_part: str
    content: str
    chunk_index: int
class BodyPart(BaseModel):
    body_part_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    part_name: str

class PostgreImageInternal(PostgreImage):
    embeddings: dict[str, list[float]] = Field(default_factory=dict)

class PostgreTextInternal(PostgreText):
    embeddings: dict[str, list[float]] = Field(default_factory=dict)