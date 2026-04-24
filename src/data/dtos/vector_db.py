from pydantic import BaseModel
from src.data.dtos.postgredb import (
    PostgreImageBase,
    PostgreImageBaseInternal,
    PostgreText,
    PostgreTextInternal,
)

class SimilaritySearchResultBase(BaseModel):
    certainty: float
    distance: float

class AgenticImagesSemanticSearchResult(SimilaritySearchResultBase):
    image: PostgreImageBase | PostgreImageBaseInternal

class AgenticTextsSemanticSearchResult(SimilaritySearchResultBase):
    text: PostgreTextInternal | PostgreText
    