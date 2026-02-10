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
        out_path: str | Path = "/home/xiangqi/xiangqi/agenticRag/src/data",
        agenticDB_data_root: str | Path = "/home/xiangqi/xiangqi/agenticRag/data/agenticDB",
        agentic_ml_url: str = "http://localhost:8000",


):
    pass