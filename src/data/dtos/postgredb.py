from pydantic import BaseModel, Field
import uuid

class PostgreFile(BaseModel):
    file_name_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    file_name: str

class PostgreImage(BaseModel):
    image_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    image_name: str
    page_number: int
    image_index: int
    image_path: str
    file_name_id: uuid.UUID
    description: str
    body_part_id: uuid.UUID
    base64_image: bytes

class BodyPart(BaseModel):
    body_part_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    part_name: str