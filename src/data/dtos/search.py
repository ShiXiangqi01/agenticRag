from pydantic import BaseModel, Field

class SimilaritySearchQuery(BaseModel):
    query: str = Field(
        description="The query string if it is a text search, or the base64 encoded image if it is an image search."
    )
    top_k: int = Field(default=5, description="The number of results to return.")
    confidence_threshold: float = Field(default=0.0, description="The minimum confidence threshold.")
    collection_names: list[str] | None = Field(
        None,
        description="The names of the collections to search in. If None, search in all collections. Only used for image searches.",
    )

class LexicalSearchQueryBase(BaseModel):
    query: str = Field(description="The query string.")
    top_k: int = Field(default=10, description="The number of results to return.")


class TextsLexicalSearchQuery(LexicalSearchQueryBase):
    search_in_body_part: bool = Field(default=True, description="Search in the body part.")
    search_in_file_name: bool = Field(default=True, description="Search in the file name.")


class ImagesLexicalSearchQuery(LexicalSearchQueryBase):
    search_in_image_name: bool = Field(default=True, description="Search in the name of the image.")
    search_in_body_part: bool = Field(default=True, description="Search in the body part.")
    search_in_file_name: bool = Field(default=True, description="Search in the file name.")
