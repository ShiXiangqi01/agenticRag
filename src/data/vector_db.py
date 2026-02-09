from asyncio.log import logger
from dtos.postgredb import (
    PostgreFile,
    PostgreImage,
    BodyPart
)
from singleton_meta import SingletonMeta
from utils import (
    load_agenticDB_images_df,
    load_agentic_images_embedding_df
)
import pandas as pd


class VectorDB(metaclass=SingletonMeta):
    
    def __init__(self):
        self._config = load_config()
        self._client = self._connect_to_weaviate()
        self._agentic_ml_client = AgenticMLClient(self._config.agentic.ml_url)
        self._query_rewriter = QueryRewriter()
        self._user_image_store = UserImageStore()

        self._images_df = load_agenticDB_images_df(
            self._config.data.images_df_file,
            self._config.app.dev_mode,
        )

        self._import_agenticDB_data(
            self._images_df
        )

        
    def _import_agenticDB_data(self, images_df: pd.DataFrame)-> None:
        
        if self._config.app.reset_vdb_on_startup:
            self._delete_all_data()
        if not self.is_initialized():
            logger.info("Importing AgenticDB data!")
            images_embedding_df = load_agentic_images_embedding_df(
                self._images_df,
                self._config.data.images_embedding_df_file,
                self._config.app.dev_mode,
            )

            self._import_agentic_images(images_df, images_embedding_df)
            logger.info("AgenticDB data import complete!")
        else:
            logger.info("AgenticDB data already initialized. Skipping import.")
        

        
    def __del__(self):
    

    def _get_client(self) -> weaviate.WeaviateClient:
        
    
    def close(self):
    
    def _connect_to_weaviate(self, raise_on_error: bool = False) -> weaviate.WeaviateClient:

    def is_initialized(self) -> bool:
    



    