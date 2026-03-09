from fastapi import APIRouter, HTTPException

from src.data.dtos.vector_db import (
    AgenticImagesSemanticSearchResult,
    AgenticTextsSemanticSearchResult,
)
from src.data.dtos.search import (
    SimilaritySearchQuery,
)
from src.data.vector_db import VectorDB
from src.ml.client import AgenticMLClient
from sentence_transformers import SentenceTransformer

router = APIRouter(prefix= "/data/search", tags=["data/search"])

vdb = VectorDB()
mlc = AgenticMLClient()
model = SentenceTransformer("BAAI/bge-m3")

@router.post(
    "/Images/similar/i2i",
    response_model = list[AgenticImagesSemanticSearchResult],
    summary= "Perform a similarity search for images via a query image.",
)
def agentic_images_similarity_search_i2i(query: SimilaritySearchQuery): 
    try:
        query_embedding = mlc.compute_image_embedding(base64_image=query.query, return_tensor="np").tolist()
        return vdb._agentic_images_img_similarity_search(
            query_embedding=query_embedding,
            top_k=query.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post(
    "/Images/similar/t2i",
    response_model = list[AgenticImagesSemanticSearchResult],
    summary= "Perform a similarity search for images via a text query.",
)
def agentic_images_similarity_search_t2i(query: SimilaritySearchQuery):
    try:
        query_embedding = mlc.compute_text_embedding(text=query.query, return_tensor="np").tolist()
        return vdb._agentic_images_description_similarity_search(
            query_embedding=query_embedding,
            top_k=query.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post(
    "/Texts/similar/t2t",
    response_model = list[AgenticTextsSemanticSearchResult],
    summary= "Perform a similarity search for texts via a text query.",
)
def agentic_texts_similarity_search_t2t(query: SimilaritySearchQuery):
    try:
        query_embedding = model.encode(query.query).tolist()
        return vdb._agentic_texts_similarity_search(
            query_embedding=query_embedding,
            top_k=10,
            target_vector="text_content",
            return_internal_texts=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
