from enum import Enum, unique
import json
from multiprocessing import Pool
from pathlib import Path
import uuid

from fire import Fire
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import pypdf
from loguru import logger
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

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

class TextEmbeddingSchema(BaseEmbeddingDataFrameSchema):
    doc_id: list[str] = Field(default_factory=list)
    chunk_index: list[int] = Field(default_factory=list)

def _read_pdf_file(file_path: Path) -> str:

    try:        
        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = pypdf.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"读取PDF文件 {file_path} 失败: {e}")
        return ""

def _load_dataframes(root: Path) -> pd.DataFrame:

    texts_df = pd.read_parquet(root / "texts.parquet")
    print(f"Loaded {len(texts_df)} records from texts.parquet")

    return texts_df

def _chunk_text_by_sentences(
        text: str, 
        sentences_per_chunk: int = 5
) -> list[str]:
    
    import re
    sentences = re.split(r'([。！？.!?])', text)
    sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2])]
    
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = ''.join(sentences[i:i + sentences_per_chunk])
        if chunk.strip():
            chunks.append(chunk.strip())
    
    return chunks
    


def _process_document(doc_item: dict, file_name: str, body_part: str) -> dict:
    content = _read_pdf_file(Path("/home/xiangqi/xiangqi/agenticRag/Database/pdf/" + doc_item.get("character_path", "")))
    if not content:
        logger.warning(f"文档内容为空: {doc_item.get('character_path', '')}")

    chunks = _chunk_text_by_sentences(content, 5)
    records = []
    for idx, chunk in enumerate(chunks):

        vector_id = str(uuid.uuid4())

        record = {
            "vector_id": vector_id,
            "doc_id": doc_item.get("doc_id", str(uuid.uuid4())),
            "character_name": doc_item.get("character_name", ""),
            "character_path": doc_item.get("character_path", ""),
            "file_name": file_name,
            "body_part": body_part,
            "content": chunk,
            "chunk_index": idx,
        }

        records.append(record)

    return records


def _create_texts_df(
        files_path: Path,
        body_part_path: Path,
        texts_data_path: Path,
        n_proc: int = 4,
) -> pd.DataFrame:
    """
    Create a DataFrame containing texts metadata from raw data sources.
    
    Args:
        files_path: Directory containing JSON files with file metadata
        body_part_path: Directory containing JSON files with body part metadata
        texts_data_path: Directory containing JSON files with text metadata
        n_proc: Number of processes for parallel processing
    
    Returns:
        DataFrame with columns: "doc_id","character_name","character_path","file_name","body_part",
                                "content","chunk_index"
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
    
    with open(texts_data_path, encoding='utf-8') as f:
        texts_data_json_list = json.load(f)
        if isinstance(texts_data_json_list["texts_records"], list) is False:
            texts_data_json_list = [texts_data_json_list["texts_records"]]
    texts_data = texts_data_json_list["texts_records"]

    records = []

    for pdf in texts_data:

        file_name = files_dict[pdf["file_name_id"]]
        body_part = body_part_dict[pdf["body_part_id"]]

        records += _process_document(pdf, file_name, body_part)
        
    
    df = pd.DataFrame(records)
    print(f"Created texts DataFrame with {len(df)} records")
    
    return df

def _generate_texts_embeddings_bgem3(
        texts_df: pd.DataFrame,
        model_name: str,
        batch_size: int,
        worker_id: int = 0,
) -> pd.DataFrame:
    model = SentenceTransformer(model_name)
    data = TextEmbeddingSchema()
    

    batch = []    
    for _, row in tqdm(
        texts_df.iterrows(),
        total = len(texts_df),
        desc="生成文本embeddings (BGE-M3)",
        position=worker_id,
        leave=False,
    ):
        batch.append(row.to_dict())
        if len(batch) < batch_size:
            continue

        # 获取embeddings
        embeddings = model.encode([row['content'] for row in batch], convert_to_numpy=True)
        # 保存结果
        for row, embedding in zip(batch, embeddings):
            data.doc_id.append(row['doc_id'])
            data.vectorDB_id.append(row['vector_id'])
            data.chunk_index.append(row['chunk_index'])
            data.embedding_type.append(EmbeddingType.TEXT)
            data.embedding_name.append(EmbeddingName.TEXT_CONTENT)
            data.embedding_model.append(model_name)
            data.embedding.append(embedding.tolist())
        batch = []
    
    return pd.DataFrame(data.model_dump())

def _generate_texts_embeddings(
        texts_df: pd.DataFrame,
        model_name: str,
        batch_size: int ,
        n_proc: int,
) -> pd.DataFrame:
    if n_proc == 1:
        return _generate_texts_embeddings_bgem3(texts_df, model_name, batch_size)
    else:
        total_rows = len(texts_df)
        idx_splits = np.array_split(np.arange(total_rows), n_proc)
        splits = [texts_df.iloc[idx] for idx in idx_splits]
        print(type(splits[0]))
        with Pool(n_proc) as pool:
            text_embeddings = pool.starmap(
                _generate_texts_embeddings_bgem3,
                [(split, model_name, batch_size, wid) for wid, split in enumerate(splits)],
            )
        text_embeddings = pd.concat(text_embeddings, ignore_index=True)
        return text_embeddings

def _generate_dataframes(
        agenticDB_data_root: Path,
        out_path: Path,
        n_proc: int,
):
    text_data_path = agenticDB_data_root/ "texts_records.json"
    files_path = agenticDB_data_root / "files.json"
    body_part_path = agenticDB_data_root / "body_part.json"
    assert agenticDB_data_root.exists(), f"{agenticDB_data_root} does not exist"

    texts_df = _create_texts_df(
        files_path,
        body_part_path,
        text_data_path,
        n_proc,
    )
    texts_df.to_parquet(out_path / "texts.parquet", index=False)
    print(f"Wrote {len(texts_df)} records to texts.parquet")

    
def _generate_embeddings(
        out_path: Path, 
        model_name: str,
        batch_size: int,
        gen_texts_embeddings: bool,
        n_proc: int,
):
    texts_df = _load_dataframes(out_path)
    if gen_texts_embeddings:
        texts_embeddings = _generate_texts_embeddings(
            texts_df,
            model_name,
            batch_size,
            n_proc,
        )
        texts_embeddings.to_parquet(out_path / "agentic_texts_embeddings.parquet", index=False)
        print(f"Wrote {len(texts_embeddings)} records to agentic_texts_embeddings.parquet")



def main(
        out_path: str | Path = "/home/xiangqi/xiangqi/agenticRag/src/data",
        agenticDB_data_root: str | Path = "/home/xiangqi/xiangqi/agenticRag/src/data/jsons",
        model_name:str = "BAAI/bge-m3",
        batch_size: int = 25,
        gen_dataframes: bool = False,
        gen_texts_embeddings: bool = True,
        n_proc: int = 1,
):
    agenticDB_data_root = Path(agenticDB_data_root)
    out_path = Path(out_path)
    if not out_path.exists():
        raise FileNotFoundError(f"{out_path} does not exist")
    
    if gen_dataframes:
        _generate_dataframes(
            agenticDB_data_root,
            out_path,
            n_proc,
        )

    if gen_texts_embeddings:
        _generate_embeddings(
            out_path,
            model_name,
            batch_size,
            gen_texts_embeddings,
            n_proc,
        )




if __name__ == "__main__":
    Fire(main)