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

    def __str__(self) :
        return self.value


class BaseEmbeddingDataFrameSchema(BaseModel):
    vectorDB_id: list[str] = Field(default_factory=list)
    embedding_type: list[EmbeddingType] = Field(default_factory=list)
    embedding_name: list[EmbeddingName] = Field(default_factory=list)
    embedding_model: list[str] = Field(default_factory=list)
    embedding:list[list[float]] = Field(default_factory=list)

class ImagesEmbeddingDataFrameSchema(BaseEmbeddingDataFrameSchema):
    postgreDB_id: list[int] = Field(default_factory=list)

def _load_dataframes(root: Path) -> pd.DataFrame:

    images_df = pd.read_parquet(root / "agentic_images.parquet")
    print(f"Loaded {len(images_df)} records from agentic_images.parquet")

    return images_df

def _generate_images_img_embeddings(
        images_df: pd.DataFrame,
        agentic_ml_url: str,
        batch_size: int,
        worker_id: int = 0,
) -> pd.DataFrame:
    data = ImagesEmbeddingDataFrameSchema()
    client = AgenticMLClient(agentic_ml_url)

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

        image_bytes = []
        for b in batch:
            try:
                image_path = Path(b["image_path"])
                image_bytes.append(read_image_bytes(image_path))
            except Exception as e:
                print(f"could not read image from {image_path}: {e}")
        
        image_inputs = EmbeddingsInput(input_data = image_bytes, input_type="image")
        image_embedding_outputs: EmbeddingsOutput = client._get_embeddings(
            image_inputs,
            None,
            False,
        )

        for b, embedding in zip(batch, image_embedding_outputs.embeddings):
            data.postgreDB_id.append(b["image_id"])
            data.vectorDB_id.append(b["vecdb_id"])
            data.embedding_type.append(EmbeddingType.IMAGE)
            data.embedding_name.append(EmbeddingName.IMAGES_IMG)
            data.embedding_model.append(image_embedding_outputs.embedding_model)
            data.embedding.append(embedding)

        batch = []
    
    return pd.DataFrame(data.model_dump())


def _generate_images_text_embeddings(
        images_df: pd.DataFrame,
        agentic_ml_url: str,
        batch_size: int,
        worker_id: int = 0,
) -> pd.DataFrame:
    data = ImagesEmbeddingDataFrameSchema()
    client = AgenticMLClient(agentic_ml_url)

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

        descriptions = [b["description"] for b in batch]
        text_inputs = EmbeddingsInput(input_data = descriptions, input_type="text")
        text_embedding_outputs: EmbeddingsOutput = client._get_embeddings(
            text_inputs,
            None,
            False,
        )

        for b, embedding in zip(batch, text_embedding_outputs.embeddings):
            data.postgreDB_id.append(b["image_id"])
            data.vectorDB_id.append(b["vecdb_id"])
            data.embedding_type.append(EmbeddingType.TEXT)
            data.embedding_name.append(EmbeddingName.IMAGES_DESCRIPTION)
            data.embedding_model.append(text_embedding_outputs.embedding_model)
            data.embedding.append(embedding)

        batch = []
    
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
        splits = np.array_split(images_df, n_proc)
        with Pool(n_proc) as pool:
            image_embeddings = pool.starmap(
                _generate_images_img_embeddings,
                [(split, agentic_ml_url, batch_size, wid) for wid, split in enumerate(splits)],
            )
        image_embeddings = pd.concat(image_embeddings)

        with Pool(n_proc) as pool:
            descriptions_embeddings = pool.starmap(
                _generate_images_text_embeddings,
                [(split, agentic_ml_url, batch_size, wid) for wid, split in enumerate(splits)],
            )
        descriptions_embeddings = pd.concat(descriptions_embeddings)

        embeddings = pd.concat([image_embeddings, descriptions_embeddings], axis=1)

    return embeddings

def _create_images_df(
        files_dir: Path,
        body_part_dir: Path,
        images_data_dir: Path,
        images_pix_dir: Path,
        n_proc: int = 4,
) -> pd.DataFrame:
    """
    Create a DataFrame containing image metadata from raw data sources.
    
    Args:
        files_dir: Directory containing JSON files with file metadata
        body_part_dir: Directory containing JSON files with body part metadata
        images_data_dir: Directory containing JSON files with image metadata
        images_pix_dir: Directory containing actual image files
        n_proc: Number of processes for parallel processing
    
    Returns:
        DataFrame with columns: image_id, image_name, page_number, image_index,
                               image_path, file_name_id, description, body_part_id,
                               base64_image, vecdb_id (UUID)
    """
    
    # Load file metadata
    files_dict = {}  # {filename: file_name_id}
    files_json_list = list(files_dir.glob("*.json"))
    for file_json in files_json_list:
        with open(file_json, 'r', encoding='utf-8') as f:
            file_data = json.load(f)
            if isinstance(file_data, list):
                file_data = file_data[0] if file_data else {}
            file_name = file_data.get("file_name", file_json.stem)
            file_id = str(uuid.uuid4())
            files_dict[file_name] = file_id
    
    # Load body part metadata
    body_part_dict = {}  # {part_name: body_part_id}
    body_part_json_list = list(body_part_dir.glob("*.json"))
    for part_json in body_part_json_list:
        with open(part_json, 'r', encoding='utf-8') as f:
            part_data = json.load(f)
            if isinstance(part_data, list):
                part_list = part_data
            else:
                part_list = [part_data]
            
            for part_item in part_list:
                part_name = part_item.get("part_name", part_json.stem)
                part_id = str(uuid.uuid4())
                body_part_dict[part_name] = part_id
    
    # Load image metadata
    images_data = []
    images_json_list = list(images_data_dir.glob("*.json"))
    
    for img_json in tqdm(images_json_list, desc="Loading image metadata"):
        with open(img_json, 'r', encoding='utf-8') as f:
            img_data = json.load(f)
            if isinstance(img_data, list):
                images_data.extend(img_data)
            else:
                images_data.append(img_data)
    
    # Build DataFrame
    records = []
    
    for img_item in tqdm(images_data, desc="Processing images"):
        image_id = str(uuid.uuid4())
        image_name = img_item.get("image_name", "")
        page_number = img_item.get("page_number", 0)
        image_index = img_item.get("image_index", 0)
        image_path = img_item.get("image_path", "")
        file_name = img_item.get("file_name", "")
        description = img_item.get("description", "")
        body_part_id = img_item.get("part", "")
        
        # Get or create file_name_id
        file_name_id = files_dict.get(file_name, str(uuid.uuid4()))
        if file_name and file_name not in files_dict:
            files_dict[file_name] = file_name_id
        
        # Get or create body_part_id
        body_part_id = body_part_dict.get(part_name, str(uuid.uuid4()))
        if part_name and part_name not in body_part_dict:
            body_part_dict[part_name] = body_part_id
        
        # Load image and convert to base64
        base64_image = None
        try:
            image_file_path = images_pix_dir / Path(image_path).name
            if not image_file_path.exists():
                # Try original path
                image_file_path = Path(image_path)
            
            if image_file_path.exists():
                base64_image = read_image_bytes(image_file_path)
        except Exception as e:
            print(f"Warning: Could not load image {image_path}: {e}")
        
        # Generate unique ID for vector DB (used in embedding steps)
        vecdb_id = str(uuid.uuid4())
        
        record = {
            "image_id": image_id,
            "image_name": image_name,
            "page_number": page_number,
            "image_index": image_index,
            "image_path": str(image_path),
            "file_name_id": file_name_id,
            "description": description,
            "body_part_id": body_part_id,
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
    images_data_dir = agenticDB_data_root / "images_data"
    images_pix_dir = agenticDB_data_root / "images_pix"
    files_dir = agenticDB_data_root / "files"
    body_part_dir = agenticDB_data_root / "body_part_data"
    assert agenticDB_data_root.exists(), f"{agenticDB_data_root} does not exist"
    assert images_data_dir.exists(), f"{images_data_dir} does not exist"
    assert images_pix_dir.exists(), f"{images_pix_dir} does not exist"  
    assert files_dir.exists(), f"{files_dir} does not exist"
    assert body_part_dir.exists(), f"{body_part_dir} does not exist"

    images_df= _create_images_df(
        files_dir,
        body_part_dir,
        images_data_dir,
        images_pix_dir,
        n_proc,
    )
    images_df.to_parquet(out_path / "images.parquet", index=False)
    print(f"Wrote {len(images_df)} records to images.parquet")






    

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
        images_embeddings.to_parquet(out_path / "agentic_images_embeddings.parquet", index=False)
        print(f"Wrote {len(images_embeddings)} records to agentic_images_embeddings.parquet")


def main(
        out_path: str | Path = "src\data\body_part.json",
        agenticDB_data_root: str | Path = "",
        agentic_ml_url: str = "http://localhost:8000",
        batch_size: int = 128,
        gen_dataframes: bool = True,
        gen_images_embeddings: bool = True,
        n_proc: int = 4,
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
    
