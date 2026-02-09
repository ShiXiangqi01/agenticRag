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

def load_agenticDB_images_df(agentic_images_df_file:str | Path, dev_mode:bool=False) -> pd.DataFrame:
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
            images_df.groupby("body_part_id")
            .sample(n = 20, random_state=42, replace=True)
            .reset_index(drop=True)
            .drop_duplicates()
        )
        logger.info(f"Using a sample of {len(images_df)} records !")
    
    images_df["image_id"] = images_df["image_id"].astype(str)
    images_df = images_df.map(array_to_list)

    return images_df

# def load_agentic_images_embedding_df(
#     images_df: pd.DataFrame,
#     agentic_images_embedding_df_file:str | Path,
#     dev_mode:bool=False
# ) -> pd.DataFrame:
     

def array_to_list(arr) -> list:
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return arr