import pandas as pd
import numpy as np
from multiprocessing import Pool
from pathlib import Path
from tqdm.auto import tqdm
from pydantic import BaseModel, Field
from enum import Enum, unique
from data.utils import read_image_bytes
from ml.client import AgenticMLClient
from ml.dto import EmbeddingsInput, EmbeddingsOutput
from fire import Fire ###############################################
import json
import uuid 

@unique
class EmbeddingType(str, Enum):
    IMAGE = "image"
    TEXT = "text"

    def __str__(self) :
        return self.value
    
@unique
class EmbeddingName(str, Enum):
    IMAGES_IMG = "images_img"
    IMAGES_DESCRIPTION = "images_description"
    TEXT_CONTENT = "text_content"

    def __str__(self) :
        return self.value


class BaseEmbeddingDataFrameSchema(BaseModel):
    vectorDB_id: list[str] = Field(default_factory=list)
    embedding_type: list[EmbeddingType] = Field(default_factory=list)
    embedding_name: list[EmbeddingName] = Field(default_factory=list)
    embedding_model: list[str] = Field(default_factory=list)
    embedding:list[list[float]] = Field(default_factory=list)

class ImagesEmbeddingDataFrameSchema(BaseEmbeddingDataFrameSchema):
    postgreDB_id: list[str] = Field(default_factory=list)

def _load_dataframes(root: Path) -> pd.DataFrame:

    images_df = pd.read_parquet(root / "imagess.parquet")
    print(f"Loaded {len(images_df)} records from imagess.parquet")

    return images_df

def _generate_images_img_embeddings(
        images_df: pd.DataFrame,
        agentic_ml_url: str,
        batch_size: int,
        worker_id: int = 0,
) -> pd.DataFrame:
    data = ImagesEmbeddingDataFrameSchema()
    client = AgenticMLClient(agentic_ml_url)

    def _process_batch(batch_items: list[dict]) -> None:
        if not batch_items:
            return

        valid_items = []
        image_bytes = []
        for item in batch_items:
            try:
                image_path = Path(item["image_path"])
                image_bytes.append(read_image_bytes(image_path))
                valid_items.append(item)
            except Exception as e:
                print(f"could not read image from {image_path}: {e}")

        if not image_bytes:
            return

        image_inputs = EmbeddingsInput(input_data=image_bytes, input_type="image")
        image_embedding_outputs: EmbeddingsOutput = client._get_embeddings(
            image_inputs,
            None,
            False,
        )

        for item, embedding in zip(valid_items, image_embedding_outputs.embeddings):
            data.postgreDB_id.append(item["image_id"])
            data.vectorDB_id.append(item["vecdb_id"])
            data.embedding_type.append(EmbeddingType.IMAGE)
            data.embedding_name.append(EmbeddingName.IMAGES_IMG)
            data.embedding_model.append(image_embedding_outputs.embedding_model)
            data.embedding.append(embedding)

    batch = []
    for _, row in tqdm(
        images_df.iterrows(),
        total = len(images_df),
        desc = "images_df img embeddings",
        position = worker_id,
        leave=False,
    ):
        batch.append(row.to_dict())
        if len(batch) < batch_size:
            continue

        _process_batch(batch)
        batch = []

    _process_batch(batch)
    
    return pd.DataFrame(data.model_dump())


def _generate_images_text_embeddings(
        images_df: pd.DataFrame,
        agentic_ml_url: str,
        batch_size: int,
        worker_id: int = 0,
) -> pd.DataFrame:
    data = ImagesEmbeddingDataFrameSchema()
    client = AgenticMLClient(agentic_ml_url)

    def _process_batch(batch_items: list[dict]) -> None:
        if not batch_items:
            return

        descriptions = [item["description"] for item in batch_items]
        text_inputs = EmbeddingsInput(input_data=descriptions, input_type="text")
        text_embedding_outputs: EmbeddingsOutput = client._get_embeddings(
            text_inputs,
            None,
            False,
        )

        for item, embedding in zip(batch_items, text_embedding_outputs.embeddings):
            data.postgreDB_id.append(item["image_id"])
            data.vectorDB_id.append(item["vecdb_id"])
            data.embedding_type.append(EmbeddingType.TEXT)
            data.embedding_name.append(EmbeddingName.IMAGES_DESCRIPTION)
            data.embedding_model.append(text_embedding_outputs.embedding_model)
            data.embedding.append(embedding)

    batch = []
    for _, row in tqdm(
        images_df.iterrows(),
        total = len(images_df),
        desc = "images_df text embeddings",
        position = worker_id,
        leave=False,
    ):
        batch.append(row.to_dict())
        if len(batch) < batch_size:
            continue

        _process_batch(batch)
        batch = []

    _process_batch(batch)
    
    return pd.DataFrame(data.model_dump())

def _generate_images_embeddings(
        images_df: pd.DataFrame,
        agentic_ml_url: str,
        batch_size: int,
        n_proc: int,
) -> pd.DataFrame:
    if n_proc == 1:
        image_embeddings = _generate_images_img_embeddings(images_df, agentic_ml_url, batch_size)
        descriptions_embeddings = _generate_images_text_embeddings(images_df, agentic_ml_url, batch_size)
    else:
        total_rows = len(images_df)
        idx_splits = np.array_split(np.arange(total_rows), n_proc)
        splits = [images_df.iloc[idx] for idx in idx_splits]
        print(type(splits[0]))
        with Pool(n_proc) as pool:
            image_embeddings = pool.starmap(
                _generate_images_img_embeddings,
                [(split, agentic_ml_url, batch_size, wid) for wid, split in enumerate(splits)],
            )
        image_embeddings = pd.concat(image_embeddings, ignore_index=True)

        with Pool(n_proc) as pool:
            descriptions_embeddings = pool.starmap(
                _generate_images_text_embeddings,
                [(split, agentic_ml_url, batch_size, wid) for wid, split in enumerate(splits)],
            )
        descriptions_embeddings = pd.concat(descriptions_embeddings, ignore_index=True)

    embeddings = pd.concat([image_embeddings, descriptions_embeddings], axis=0, ignore_index=True)

    return embeddings

def _create_images_df(
        files_path: Path,
        body_part_path: Path,
        images_data_path: Path,
        images_pix_path: Path,
        n_proc: int = 4,
) -> pd.DataFrame:
    """
    Create a DataFrame containing image metadata from raw data sources.
    
    Args:
        files_path: Directory containing JSON files with file metadata
        body_part_path: Directory containing JSON files with body part metadata
        images_data_path: Directory containing JSON files with image metadata
        images_pix_path: Directory containing actual image files
        n_proc: Number of processes for parallel processing
    
    Returns:
        DataFrame with columns: image_id, image_name, page_number, image_index,
                               image_path, file_name_id, description, body_part_id,
                               base64_image, vecdb_id (UUID)
    """

    with open(files_path, encoding='utf-8') as f:
        files_json_list = json.load(f)
        if isinstance(files_json_list["files"], list) is False:
            files_json_list = [files_json_list["files"]]
    files_json_list = files_json_list["files"]

    files_dict = {}
    for file_data in files_json_list:
        
        file_name = file_data.get("file_name", "")
        file_id = file_data.get("file_name_id", str(uuid.uuid4()))
        files_dict[file_id] = file_name

    with open(body_part_path, encoding='utf-8') as f:
        body_part_json_list = json.load(f)
        if isinstance(body_part_json_list["body_part"], list) is False:
            body_part_json_list = [body_part_json_list["body_part"]]
    body_part_json_list = body_part_json_list["body_part"]

    body_part_dict = {}
    for body_part_data in body_part_json_list:
        
        part_name = body_part_data.get("part_name", "")
        part_id = body_part_data.get("body_part_id", str(uuid.uuid4()))
        body_part_dict[part_id] = part_name
    
    with open(images_data_path, encoding='utf-8') as f:
        images_data_json_list = json.load(f)
        if isinstance(images_data_json_list["image_records"], list) is False:
            images_data_json_list = [images_data_json_list["image_records"]]
    images_data = images_data_json_list["image_records"]

    records = []
    
    for img_item in tqdm(images_data, desc="Processing images"):
        image_id = img_item.get("image_id", str(uuid.uuid4()))
        image_name = img_item.get("image_name", "")
        page_number = img_item.get("page_number", 0)
        image_index = img_item.get("image_index", 0)
        image_path = img_item.get("image_path", "")
        file_name_id = img_item.get("file_name_id", "")
        description = img_item.get("description", "")
        body_part_id = img_item.get("body_part_id", "")
        base64_image = img_item.get("base64_image")
        
        # Get or create file_name_id
        
        file_name = files_dict[file_name_id]
        
        # Get or create body_part_id
        body_part = body_part_dict[body_part_id]
        
        # Use existing base64 if provided; otherwise load from file when possible
        
        
        
        # Generate unique ID for vector DB (used in embedding steps)
        vecdb_id = str(uuid.uuid4())
        
        record = {
            "image_id": image_id,
            "image_name": image_name,
            "page_number": page_number,
            "image_index": image_index,
            "image_path": str(image_path),
            "file_name": file_name,
            "description": description,
            "body_part": body_part,
            "base64_image": base64_image,
            "vecdb_id": vecdb_id,
        }
        
        records.append(record)
    
    df = pd.DataFrame(records)
    print(f"Created images DataFrame with {len(df)} records")
    
    return df
    




def _generate_dataframes(
        agenticDB_data_root: Path,
        out_path: Path,
        n_proc: int = 4,
):
    images_data_path = agenticDB_data_root / "image_recordss.json"
    # images_pix_dir = agenticDB_data_root / "images_pix"
    files_path = agenticDB_data_root / "files.json"
    body_part_path = agenticDB_data_root / "body_part.json"
    assert agenticDB_data_root.exists(), f"{agenticDB_data_root} does not exist"


    images_df= _create_images_df(
        files_path,
        body_part_path,
        images_data_path,
        None,
        n_proc,
    )
    images_df.to_parquet(out_path / "imagess.parquet", index=False)
    print(f"Wrote {len(images_df)} records to imagess.parquet")






    

def _generate_embeddings(
        out_path: Path, 
        agentic_ml_url: str,
        batch_size: int,
        gen_images_embeddings: bool,
        n_proc: int,
):  
    images_df = _load_dataframes(out_path)
    if gen_images_embeddings:
        images_embeddings = _generate_images_embeddings(
            images_df,
            agentic_ml_url,
            batch_size,
            n_proc,
        )
        images_embeddings.to_parquet(out_path / "agentic_images_embeddingss.parquet", index=False)
        print(f"Wrote {len(images_embeddings)} records to agentic_images_embeddingss.parquet")


def main(
        out_path: str | Path = "/home/xiangqi/xiangqi/agenticRag/src/data",
        agenticDB_data_root: str | Path = "/home/xiangqi/xiangqi/agenticRag/src/data/jsons",
        agentic_ml_url: str = "http://localhost:8001",
        batch_size: int = 25,
        gen_dataframes: bool = True,
        gen_images_embeddings: bool = True,
        n_proc: int = 1,
):
    agenticDB_data_root = Path(agenticDB_data_root)
    out_path = Path(out_path)
    if not out_path.exists():
        raise FileNotFoundError(f"{out_path} does not exist")
    # if gen_images_embeddings:
    #     AgenticMLClient(agentic_ml_url)
    
    if gen_dataframes:
        _generate_dataframes(
            agenticDB_data_root,
            out_path,
            n_proc,
        )

    if gen_images_embeddings:
        _generate_embeddings(
            out_path,
            agentic_ml_url,
            batch_size,
            gen_images_embeddings,
            n_proc,
        )

if __name__ == "__main__":
    Fire(main)###########################
    
