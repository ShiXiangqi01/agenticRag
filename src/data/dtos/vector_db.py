from pydantic import BaseModel
from src.data.dtos.postgredb import (
    PostgreImage,
    PostgreImageInternal,
    PostgreText,
    PostgreTextInternal,
)

class SimilaritySearchResultBase(BaseModel):
    certainty: float
    distance: float

class AgenticImagesSemanticSearchResult(SimilaritySearchResultBase):
    image: PostgreImageInternal | PostgreImage

class AgenticTextsSemanticSearchResult(SimilaritySearchResultBase):
    text: PostgreTextInternal | PostgreText
    