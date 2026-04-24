from fastapi import APIRouter, HTTPException, Path, Query

from src.data.dtos.postgredb import (
    PostgreImageBase,
    PostgreText,
)
from src.data.vector_db import VectorDB

router = APIRouter(prefix="/data/lookup", tags=["data/lookup"])
vdb = VectorDB()

@router.get(
    "/text/surroudings",
    response_model=list[PostgreText],
    summary="Find surrounding chunks for a given chunk",
)
def find_surrounding_chunks(
    chunk_vector_id: str = Query(..., description="The vector ID of the chunk"),
    top_k: int = Query(2, description="The number of surrounding chunks to return"),
):
    try:
        surrounding_chunks = vdb._find_surrounding_chunks(chunk_vector_id, top_k)
        return surrounding_chunks
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get(
    "/text",
    response_model=PostgreText,
    summary="Get a text record by its vector ID",
)
def get_text_record_by_vector_id(
    vector_id: str = Query(..., description="The vector ID of the text record"),
):
    try:
        text_record = vdb.get_text_record_by_vector_id(vector_id)
        if not text_record:
            raise HTTPException(status_code=404, detail="Text record not found")
        return text_record
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get(
    "/image",
    response_model=PostgreImageBase,
    summary="Get an image record by its vector ID",
)
def get_image_record_by_vector_id(
    vector_id: str = Query(..., description="The vector ID of the image record"),
):
    try:
        image_record = vdb.get_image_record_by_vector_id(vector_id)
        if not image_record:
            raise HTTPException(status_code=404, detail="Image record not found")
        return image_record
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get(
    "/body_parts",
    response_model=list[str],
    summary="List all body parts in the database",
)
def list_all_body_parts():
    try:
        body_parts = vdb.list_all_body_parts()
        return body_parts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/file_names",
    response_model=list[str],
    summary="List all file names in the database",
)
def list_all_file_names():
    try:
        file_names = vdb.list_all_file_names()
        return file_names
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 


@router.get(
    "/character_names/{file_name}",
    response_model=list[str],
    summary="List all character names in a given file",
)
def list_all_character_names_in_one_file(
    file_name: str = Path(..., description="The name of the file"),
):
    try:
        character_names = vdb.list_all_character_names_in_one_file(file_name)
        return character_names
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))