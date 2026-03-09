from asyncio.log import logger
import time
from typing import Any, Literal

import mlflow
from mlflow.entities import SpanType
import tqdm
import weaviate
from weaviate.classes.query import MetadataQuery
from src.data.dtos.postgredb import (
    PostgreFile,
    PostgreImage,
    PostgreImageInternal,
    PostgreText,
    PostgreTextInternal,
    BodyPart
)
from src.singleton_meta import SingletonMeta
from src.data.utils import (
    load_postgreDB_images_df,
    load_postgreDB_images_embedding_df,
    read_image_bytes,
    load_postgreDB_texts_df,
    load_postgreDB_texts_embedding_df,
)
from src.data.dtos.vector_db import (
    AgenticImagesSemanticSearchResult,
    AgenticTextsSemanticSearchResult,
)
import pandas as pd
from src.config import load_config
from src.ml.client import AgenticMLClient
from src.data.schema import (
    AGENTIC_IMAGES_SCHEMA_NAME,
    AGENTIC_IMAGES_SCHEMA_PROPS,
    AGENTIC_IMAGES_SCHEMA_VECTORIZER,
    AGENTIC_TEXTS_SCHEMA_NAME,
    AGENTIC_TEXTS_SCHEMA_PROPS,
    AGENTIC_TEXTS_SCHEMA_VECTORIZER,
)

class VectorDB(metaclass=SingletonMeta):
    
    def __init__(self):
        self._config = load_config()
        self._client = self._connect_to_weaviate()
        self._agentic_ml_client = AgenticMLClient(self._config.agentic.ml_url)
        # self._query_rewriter = QueryRewriter()
        # self._user_image_store = UserImageStore()

        self._images_df = load_postgreDB_images_df(
            self._config.data.images_df_file,
            self._config.app.dev_mode,
        )
        self._texts_df = load_postgreDB_texts_df(
            self._config.data.texts_df_file,
            self._config.app.dev_mode,
        )
        self._import_agenticDB_data(
            self._images_df,
            self._texts_df
        )

        

        
    def __del__(self):
        try:
            self.close()
        except Exception as e:
            logger.error(f"Error while closing VectorDB: {e}")
    

    def _get_client(self) -> weaviate.WeaviateClient:
        if self._client.is_ready():
            return self._client
        
        MAX_RETRIES = 5
        while not self._client.is_ready() and MAX_RETRIES > 1:
            self._client = self._connect_to_weaviate()
            MAX_RETRIES -= 1
            time.sleep(1)
        
        self._client = self._connect_to_weaviate(raise_on_error=True)
        return self._client
        
    
    def close(self):
        self._get_client().close()
    
    def _connect_to_weaviate(self, raise_on_error: bool = False) -> weaviate.WeaviateClient:
        client = weaviate.connect_to_custom(
            http_host=self._config.weaviate.host,
            http_port=self._config.weaviate.http_port,
            http_secure=False,
            grpc_host=self._config.weaviate.host,
            grpc_port=self._config.weaviate.grpc_port,
            grpc_secure=False,
        )

        if not client.is_ready():
            msg = f"Failed to connect to Weaviate at {self._config.weaviate.host}:{self._config.weaviate.http_port}"
            logger.warning(msg)
            if raise_on_error:
                raise ConnectionError(msg)
        logger.info(f"Successfully connected to Weaviate at {self._config.weaviate.host}:{self._config.weaviate.http_port}")
        return client

    def is_initialized(self) -> bool:
        return self._get_client().collections.exists(
            AGENTIC_IMAGES_SCHEMA_NAME
        ) and self._get_client().collections.exists(
            AGENTIC_TEXTS_SCHEMA_NAME
        )
 
    
    def _delete_all_data(self) -> None:
        logger.warning("Deleting all collections in Weaviate...")
        client = self._get_client()
        client.collections.delete_all()

    def _create_agentic_images_schema(self) -> weaviate.collections.Collection:
        return self._get_client().collections.create(
            name = AGENTIC_IMAGES_SCHEMA_NAME,
            properties=AGENTIC_IMAGES_SCHEMA_PROPS,  # comment out for auto schema creation
            vectorizer_config=AGENTIC_IMAGES_SCHEMA_VECTORIZER,
            
        ) and self._get_client().collections.get(AGENTIC_IMAGES_SCHEMA_NAME)
    
    def _create_agentic_texts_schema(self) -> weaviate.collections.Collection:
        return self._get_client().collections.create(
            name = AGENTIC_TEXTS_SCHEMA_NAME,
            properties=AGENTIC_TEXTS_SCHEMA_PROPS,
            vectorizer_config=AGENTIC_TEXTS_SCHEMA_VECTORIZER,
        ) and self._get_client().collections.get(AGENTIC_TEXTS_SCHEMA_NAME)

    def _create_agentic_images_from_query_results(self, res: Any) -> list[PostgreImage | PostgreImageInternal]:

        images = []
        for res_obj in res.objects:
            props = res_obj.properties
            image_internal = None
            embeddings = dict()
            image_record = PostgreImage(
                image_id = props["postgre_id"],
                vector_id = props["vector_id"],
                image_name = props["image_name"],
                page_number = props["page_number"],
                image_index = props["image_index"],
                image_path = props["image_path"],
                file_name = props["file_name"],
                description = props["description"],
                body_part = props["body_part"],
                base64_image = props["base64_image"]
            )

            embeddings = {}
            if res_obj.vector is not None:
                if isinstance(res_obj.vector, dict):
                    embeddings = res_obj.vector
                else:
                    embeddings["default"] = res_obj.vector
                image_internal = PostgreImageInternal(
                    **image_record.model_dump(),
                    embeddings=embeddings,
                )
            images.append(image_internal if image_internal else image_record)
        
        return images
            
    def _create_agentic_texts_from_query_results(self, res: Any) -> list[PostgreText | PostgreTextInternal]:

        texts = []
        for res_obj in res.objects:
            props = res_obj.properties
            text_internal = None
            embeddings = dict()
            text_record = PostgreText(
                doc_id = props["doc_id"],
                vector_id = props["vector_id"],
                character_path = props["character_path"],
                file_name = props["file_name"],
                body_part = props["body_part"],
                content = props["content"],
                chunk_index = props["chunk_index"],
            )

            embeddings = {}
            if res_obj.vector is not None:
                if isinstance(res_obj.vector, dict):
                    embeddings = res_obj.vector
                else:
                    embeddings["default"] = res_obj.vector
                text_internal = PostgreTextInternal(
                    **text_record.model_dump(),
                    embeddings=embeddings,
                )
            texts.append(text_internal if text_internal else text_record)

        return texts
    
    def _import_postgreDB_Images(
            self,
            images_df: pd.DataFrame,
            images_embedding_df: pd.DataFrame,
    )-> None:
        if self._get_client().collections.exists(AGENTIC_IMAGES_SCHEMA_NAME):
            return
        
        logger.info(f"Importing AgenticDB images data into Weaviate...")

        collection = self._create_agentic_images_schema()

        with collection.batch.fixed_size(batch_size= 25, concurrent_requests= 1) as batch:
        
            for idx, row in tqdm.tqdm(
                images_df.iterrows(),
                total = len(images_df),
                desc = "Importing AgenticDB images data into Weaviate",
                leave = True,
            ):
                img_path = (self._config.data.agentic_data_root + "/images/" + row["image_path"]).replace("//", "/")
                try:
                    base64_image = read_image_bytes(img_path)
                except Exception:
                    logger.warning(f"Failed to read image at {img_path}, skipping...")
                    continue

                props = {
                    
                    "vector_id": row["vecdb_id"],
                    "postgre_id": row["image_id"],
                    "image_name": row["image_name"],
                    "page_number": row["page_number"],
                    "image_index": row["image_index"],
                    "image_path": str(img_path),
                    "file_name": row["file_name"],
                    "description": row["description"],
                    "body_part": row["body_part"],
                    "base64_image": base64_image,

                }

                images_embeddings = images_embedding_df[images_embedding_df["vectorDB_id"] == row["vecdb_id"]]
                if len(images_embeddings) == 0:
                    vector = None
                if len(images_embeddings) == 1:
                    vector = images_embeddings.iloc[0]["embedding"]
                else:
                    vector = {emb["embedding_name"]: emb["embedding"] for _, emb in images_embeddings.iterrows()}

                batch.add_object(
                    uuid = row["vecdb_id"],
                    properties = props,
                    vector = vector,
                )

        logger.info(f"Imported AgenticDB images records into Weaviate.")
    
    def _import_agenticDB_texts(
            self,
            texts_df: pd.DataFrame,
            texts_embedding_df: pd.DataFrame,
    )-> None:
        if self._get_client().collections.exists(AGENTIC_TEXTS_SCHEMA_NAME):
            return
        
        logger.info(f"Importing AgenticDB texts data into Weaviate...")

        collection = self._create_agentic_texts_schema()

        with collection.batch.fixed_size(batch_size= 25, concurrent_requests= 1) as batch:
        
            for idx, row in tqdm.tqdm(
                texts_df.iterrows(),
                total = len(texts_df),
                desc = "Importing AgenticDB texts data into Weaviate",
                leave = True,
            ):
                props = {
                    "vector_id": row["vector_id"],
                    "doc_id": row["doc_id"],
                    "character_path": row["character_path"],
                    "file_name": row["file_name"],
                    "body_part": row["body_part"],
                    "content": row["content"],  
                    "chunk_index": row["chunk_index"],
                }

                text_embeddings = texts_embedding_df[texts_embedding_df["vectorDB_id"] == row["vector_id"]]
                if len(text_embeddings) == 0:
                    vector = None
                if len(text_embeddings) == 1:
                    vector = {text_embeddings.iloc[0]["embedding_name"]: text_embeddings.iloc[0]["embedding"]}
                else:
                    vector = {emb["embedding_name"]: emb["embedding"] for _, emb in text_embeddings.iterrows()}

                batch.add_object(
                    uuid = row["vector_id"],
                    properties = props,
                    vector = vector,
                )

        logger.info(f"Imported AgenticDB text records into Weaviate.")
    

    def _import_agenticDB_data(
        self,
        images_df: pd.DataFrame,
        texts_df: pd.DataFrame,
    ) -> None:
        
        if self._config.app.reset_vdb_on_startup:
            self._delete_all_data()
        if not self.is_initialized():
            logger.info("Importing AgenticDB data...")
            images_embeddings_df = load_postgreDB_images_embedding_df(
                images_df,
                self._config.data.images_embedding_df_file,
                self._config.app.dev_mode,
            )
            texts_embeddings_df = load_postgreDB_texts_embedding_df(
                texts_df,
                self._config.data.texts_embedding_df_file,
                self._config.app.dev_mode,
            )

            self._import_postgreDB_Images(images_df, images_embeddings_df)
            self._import_agenticDB_texts(texts_df, texts_embeddings_df)
            logger.info("AgenticDB data import complete.")
        else:
            logger.info("AgenticDB data already imported.")
        

    def _agentic_images_img_similarity_search(
            self,
            query_embedding: list[float],
            top_k: int = 5,
            return_internal_images: bool = False,
    ) -> list[AgenticImagesSemanticSearchResult]:
        """
        Perform a similarity search of 'Images' based on their image embedding.

        Args:
            query_embedding (list[float]): The query embedding vector.. 
            top_k (int): The number of top results to return.
            return_internal_images (bool): Whether to return internal image data (base64) in the results.

        Returns:
            list[AgenticImagesSemanticSearchResult]: A list of search results, each containing the image data and similarity metrics.
        """

        result = self._agentic_image_similarity_search(
            query_embedding=query_embedding,
            target_vector = "images_img",
            top_k=top_k,
            return_internal_images=return_internal_images,
        )
        return result

    def _agentic_images_description_similarity_search(
            self,
            query_embedding: list[float],
            top_k: int = 5,
            return_internal_images: bool = False,
    ) -> list[AgenticImagesSemanticSearchResult]:
        """
        Perform a similarity search of 'Images' based on their description embedding.

        Args:
            query_embedding (list[float]): The query embedding vector.. 
            top_k (int): The number of top results to return.
            return_internal_images (bool): Whether to return internal image data (base64) in the results.

        Returns:
            list[AgenticImagesSemanticSearchResult]: A list of search results, each containing the image data and similarity metrics.
        """

        result = self._agentic_image_similarity_search(
            query_embedding=query_embedding,
            target_vector = "images_description",
            top_k=top_k,
            return_internal_images=return_internal_images,
        )
        return result
    
    
    def _agentic_image_similarity_search(
            self,
            query_embedding: list[float],
            target_vector: Literal["images_img", "images_description"],
            top_k: int = 5,
            return_internal_images: bool = False,
    ) -> list[AgenticImagesSemanticSearchResult]:
        """
        Perform a similarity search of 'Images' via their image embedding or description embedding.

        Args:
            query_embedding (list[float]): The query embedding vector.. 
            target_vector (str): The target vector field to search against, either "images_img" or "images_description".
            top_k (int): The number of top results to return.
            return_internal_images (bool): Whether to return image's vector in the results.

        Returns:
            list[AgenticImagesSemanticSearchResult]: A list of search results with similarity scores
        """

        filter = None
        collection = self._get_client().collections.get("AgenticImages")

        return_props = list(PostgreImage.model_fields.keys())
        include_vector = False

        if return_internal_images:
            include_vector = ["images_img","images_description"]

        results = collection.query.near_vector(
            query_embedding,
            target_vector = target_vector,
            filters=filter,
            limit=int(top_k),
            return_metadata=MetadataQuery(certainty=True, distance=True),
            return_properties=return_props,
            include_vector=include_vector,
        )

        records = self._create_agentic_images_from_query_results(results)

        simsearch_results = []
        for image_record, res_obj in zip(records, results.objects):
            res = AgenticImagesSemanticSearchResult(
                image=image_record,
                certainty=res_obj.metadata.certainty,
                distance=res_obj.metadata.distance,
            )
            simsearch_results.append(res)
        
        return simsearch_results
    
    def _agentic_texts_similarity_search(
            self,
            query_embedding: list[float],
            target_vector: Literal["text_content"],
            top_k: int = 10,
            return_internal_texts: bool = False,
    ) -> list[AgenticTextsSemanticSearchResult]:
        """
        Perform a similarity search of 'Texts' based on their content embedding.

        Args:
            query_embedding (list[float]): The query embedding vector.. 
            target_vector (str): The target vector field to search against, currently only supports "texts_content".
            top_k (int): The number of top results to return.
            return_internal_texts (bool): Whether to return text's vector in the results.

        Returns:
            list[AgenticTextsSemanticSearchResult]: A list of search results, each containing the text data and similarity metrics.
        """
        filter = None
        collection = self._get_client().collections.get("AgenticTexts")

        return_props = list(PostgreText.model_fields.keys())
        include_vector = False

        if return_internal_texts:
            include_vector = ["text_content"]

        results = collection.query.near_vector(
            query_embedding,
            target_vector = target_vector,
            filters=filter,
            limit=int(top_k),
            return_metadata=MetadataQuery(certainty=True, distance=True),
            return_properties=return_props,
            include_vector=include_vector,
        )

        records = self._create_agentic_texts_from_query_results(results)

        simsearch_results = []
        for text_record, res_obj in zip(records, results.objects):
            res = AgenticTextsSemanticSearchResult(
                text=text_record,
                certainty=res_obj.metadata.certainty,
                distance=res_obj.metadata.distance,
            )
            simsearch_results.append(res)
        
        return simsearch_results
        
    @mlflow.trace(
        span_type=SpanType.TOOL,
    )
    def find_agentic_images_with_similar_image(                #m2m
        self,
        vector_id: str,
        search_in_collections: list[str] | None = None,
        top_k: int = 5,
    ) -> list[AgenticImagesSemanticSearchResult]:
        """
        Find `ImageRecord`s with images similar to the one specified by the `vector_id`. This uses semantic similarity search based on image embeddings.

        Args:
            vector_id (str): The unique identifier of the image in the VectorDB.
            search_in_collections (list[str], optional): Names of `FundusCollection`s to restrict the search. Defaults to None.
            top_k (int, optional): Number of top results to return. Defaults to 10

        Returns:
            list[AgenticImagesSemanticSearchResult]: `ImageSimilarity`s search results with similarity scores.
        """
        record = self.get_agentic_image_internal_by_vector_id(vector_id)
        image_embedding = self._agentic_ml_client.compute_image_embedding(
            record.base64_image, return_tensor="np"
        ).tolist()  # type: ignore
        results = self._agentic_images_img_similarity_search(
            image_embedding,
            search_in_collections=search_in_collections,
            top_k=top_k,
        )
        return results
    
    @mlflow.trace(
        span_type=SpanType.TOOL,
    )
    def find_agentic_images_with_img_similar_to_the_text_query(                  #t2m
        self,
        query:str,
        search_in_collections: list[str] | None = None,
        top_k: int = 5,
    ) -> list[AgenticImagesSemanticSearchResult]:
        """
        Find `ImageRecord`s with images similar to the text query. This uses semantic similarity search based on image embeddings.

        Args:
            query (str): The text query to find similar images for.
            search_in_collections (list[str], optional): Names of `FundusCollection`s to restrict the search. Defaults to None.
            top_k (int, optional): Number of top results to return. Defaults to 10

        Returns:
            list[AgenticImagesSemanticSearchResult]: `ImageSimilarity`s search results with similarity scores.
        """
        # query = self._query_rewriter.rewrite_user_query_for_cross_modal_text_to_image_search(query)
        query_embedding = self._agentic_ml_client.compute_text_embedding(
            query, return_tensor="np"
        ).tolist()  # type: ignore
        results = self._agentic_images_img_similarity_search(
            query_embedding,
            search_in_collections=search_in_collections,
            top_k=top_k,
        )
        return results

    @mlflow.trace(
        span_type=SpanType.TOOL,
    )
    def find_agentic_images_with_img_similar_to_user_image(          #m2m
        self,
        user_image_id: str,
        search_in_collections: list[str] | None = None,
        top_k: int = 5,
    )-> list[AgenticImagesSemanticSearchResult]:
        """
        Find `ImagesRecord`s with images similar to the user-provided image.
        This uses image similarity search based on image embeddings.
        Use this if a user provides an image and you want to find similar images in the database.

        Args:
            user_image_id (str): The unique identifier of the user-provided image.
            search_in_collections (list[str], optional): Names of `FundusCollection`s to restrict the search. Defaults to None.
            top_k (int, optional): Number of top results to return. Defaults to 10

        Returns:
            list[AgenticImagesSemanticSearchResult]: `ImageSimilarity`s search results with similarity scores.
        """
        # base64_user_image: str = self._user_image_store.load_user_image(user_image_id, base64=True)  # type: ignore
        # base64_user_image = base64_user_image.split(",")[-1]
        image_embedding = self._agentic_ml_client.compute_image_embedding(
            base64_image= base64_user_image, return_tensor="np" 
        ).tolist()  # type: ignore
        results = self._agentic_images_img_similarity_search(
            image_embedding,
            search_in_collections=search_in_collections,
            top_k=top_k,
        )
        return results
    
    @mlflow.trace(
        span_type=SpanType.TOOL,
    )
    def find_agentic_images_with_description_similar_to_the_text_query(          #t2t
        self,
        query: str,
        search_in_collections: list[str] | None = None,
        top_k: int = 5,
    ) -> list[AgenticImagesSemanticSearchResult]:
        """
        Find `ImageRecord`s with image descriptions similar to the text query.
        The uses textual semantic similarity search based on description embeddings.
        Use this only to search for imagesRecord based on the similarity between the text query and the image descriptions in the database.
        if you want to search for images record based on their images, use `find_agentic_images_with_similar_image` instead.

        Args:
            query (str): The text query to find similar image descriptions for.
            search_in_collections (list[str], optional): Names of `FundusCollection`s to restrict the search. Defaults to None.
            top_k (int, optional): Number of top results to return. Defaults to 10

        Returns:
            list[AgenticImagesSemanticSearchResult]: `ImageSimilarity`s search results with similarity scores.
        """
        # query = self._query_rewriter.rewrite_user_query_for_cross_modal_text_to_image_search(query)
        query_embedding = self._agentic_ml_client.compute_text_embedding(
            query, return_tensor="np"
        ).tolist()  # type: ignore
        results = self._agentic_images_desc_similarity_search(
            query_embedding,
            search_in_collections=search_in_collections,
            top_k=top_k,
        )
        return results
    
    



    