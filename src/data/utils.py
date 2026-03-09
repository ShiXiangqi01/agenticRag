from pathlib import Path
from loguru import logger
from PIL import Image
import base64
import io

import pandas as pd
import numpy as np

def read_image_bytes(image_path: Path) -> str:
    try:
        image = Image.open(image_path).convert("RGB")
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        raise FileNotFoundError(f"Could not read image from {image_path}: {e}")

def load_postgreDB_images_df(agentic_images_df_file:str | Path, dev_mode:bool=False) -> pd.DataFrame:
    agentic_images_df_file = Path(agentic_images_df_file)
    if not agentic_images_df_file.exists():
        raise FileNotFoundError(f"Agentic images dataframe file not found: {agentic_images_df_file.absolute()}")
    logger.info(f"Loading Agentic images dataframe from {agentic_images_df_file}...")
    images_df = pd.read_parquet(agentic_images_df_file)
    logger.info(f"Loaded agentic images with {len(images_df)} records.")

    if dev_mode:
        logger.info("Using DataFrame in dev mode!")
        '''
        这段代码把 records_df 按 collection_name 分组后，每组随机抽取 20 条（固定随机种子 42、允许放回抽样），再重置索引并去重，得到一个更小的样本数据集。
        '''
        images_df = (
            images_df.groupby("body_part")
            .sample(n = 20, random_state=42, replace=True)
            .reset_index(drop=True)
            .drop_duplicates()
        )
    logger.info(f"Using a sample of {len(images_df)} records !")
    
    images_df["image_id"] = images_df["image_id"].astype(str)
    images_df = images_df.map(array_to_list)
    

    return images_df

def load_postgreDB_images_embedding_df(
    images_df: pd.DataFrame,
    postgreDB_images_embedding_df_file:str | Path,
    dev_mode:bool=False
) -> pd.DataFrame:
    postgreDB_images_embedding_df_file = Path(postgreDB_images_embedding_df_file)
    if not postgreDB_images_embedding_df_file.exists():
        raise FileNotFoundError(f"PostgreDB images embedding dataframe file not found: {postgreDB_images_embedding_df_file.absolute()}")
    logger.info(f"Loading PostgreDB images embedding dataframe from {postgreDB_images_embedding_df_file}...")
    images_embedding_df = pd.read_parquet(postgreDB_images_embedding_df_file)
    logger.info(f"Loaded PostgreDB images embedding with {len(images_embedding_df)} records.")

    if dev_mode:
        if images_df is None:
            raise ValueError("Cannot load images embedding dataframe in dev mode without loading the records dataframe first.")
        logger.info("Using PostgreDB images embedding dataframe in dev mode!")
        images_embedding_df = images_embedding_df[images_embedding_df["postgre_id"].isin(images_df["image_id"])]
        logger.info(f"Using a sample of {len(images_embedding_df)} records !")


    return images_embedding_df

def load_postgreDB_texts_df(agentic_texts_df_file:str | Path, dev_mode:bool=False) -> pd.DataFrame:
    agentic_texts_df_file = Path(agentic_texts_df_file)
    if not agentic_texts_df_file.exists():
        raise FileNotFoundError(f"Agentic texts dataframe file not found: {agentic_texts_df_file.absolute()}")
    logger.info(f"Loading Agentic texts dataframe from {agentic_texts_df_file}...")
    texts_df = pd.read_parquet(agentic_texts_df_file)
    logger.info(f"Loaded agentic texts with {len(texts_df)} records.")

    if dev_mode:
        logger.info("Using DataFrame in dev mode!")
        '''
        这段代码把 records_df 按 collection_name 分组后，每组随机抽取 20 条（固定随机种子 42、允许放回抽样），再重置索引并去重，得到一个更小的样本数据集。
        '''
        texts_df = (
            texts_df.groupby("doc_id")
            .sample(n = 20, random_state=42, replace=True)
            .reset_index(drop=True)
            .drop_duplicates()
        )
    logger.info(f"Using a sample of {len(texts_df)} records !")
    
    texts_df["doc_id"] = texts_df["doc_id"].astype(str)
    texts_df = texts_df.map(array_to_list)
    
    return texts_df

def load_postgreDB_texts_embedding_df(
    texts_df: pd.DataFrame,
    postgreDB_texts_embedding_df_file:str | Path,
    dev_mode:bool=False
) -> pd.DataFrame:
    postgreDB_texts_embedding_df_file = Path(postgreDB_texts_embedding_df_file)
    if not postgreDB_texts_embedding_df_file.exists():
        raise FileNotFoundError(f"PostgreDB texts embedding dataframe file not found: {postgreDB_texts_embedding_df_file.absolute()}")
    logger.info(f"Loading PostgreDB texts embedding dataframe from {postgreDB_texts_embedding_df_file}...")
    texts_embedding_df = pd.read_parquet(postgreDB_texts_embedding_df_file)
    logger.info(f"Loaded PostgreDB texts embedding with {len(texts_embedding_df)} records.")

    if dev_mode:
        if texts_df is None:
            raise ValueError("Cannot load texts embedding dataframe in dev mode without loading the records dataframe first.")
        logger.info("Using PostgreDB texts embedding dataframe in dev mode!")
        texts_embedding_df = texts_embedding_df[texts_embedding_df["doc_id"].isin(texts_df["doc_id"])]
        logger.info(f"Using a sample of {len(texts_embedding_df)} records !")

    return texts_embedding_df    

def array_to_list(arr) -> list:
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return arr