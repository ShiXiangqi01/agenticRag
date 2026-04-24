from asyncio.log import logger
import time
from typing import Any, Literal

import mlflow
from mlflow.entities import SpanType
import tqdm
import weaviate
from weaviate.classes.query import Filter, MetadataQuery
from src.data.dtos.postgredb import (
    PostgreFile,
    PostgreImage,
    PostgreImageBase,
    PostgreImageBaseInternal,
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
from sentence_transformers import SentenceTransformer
from src.data.schema import (
    AGENTIC_IMAGES_SCHEMA_NAME,
    AGENTIC_IMAGES_SCHEMA_PROPS,
    AGENTIC_IMAGES_SCHEMA_VECTORIZER,
    AGENTIC_TEXTS_SCHEMA_NAME,
    AGENTIC_TEXTS_SCHEMA_PROPS,
    AGENTIC_TEXTS_SCHEMA_VECTORIZER,
)
from src.data.user_image_store import UserImageStore
from src.agent.tools.query_rewriter import QueryRewriter

class VectorDB(metaclass=SingletonMeta):
    
    def __init__(self):
        self._config = load_config()
        self._client = self._connect_to_weaviate()
        self._agentic_ml_client = AgenticMLClient(self._config.agentic.ml_url)
        self._model = SentenceTransformer(self._config.agentic.model_name)
        self._query_rewriter = QueryRewriter()
        self._user_image_store = UserImageStore()

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

    def _create_agentic_images_from_query_results(self, res: Any) -> list[PostgreImageBase | PostgreImageBaseInternal]:

        images = []
        for res_obj in res.objects:
            props = res_obj.properties
            image_internal = None
            embeddings = dict()
            image_record = PostgreImageBase(
                image_id = props["postgre_id"],
                vector_id = props["vector_id"],
                image_name = props["image_name"],
                page_number = props["page_number"],
                image_index = props["image_index"],
                image_path = props["image_path"],
                file_name = props["file_name"],
                description = props["description"],
                body_part = props["body_part"],
            )

            embeddings = {}
            if res_obj.vector is not None:
                if isinstance(res_obj.vector, dict):
                    embeddings = res_obj.vector
                else:
                    embeddings["default"] = res_obj.vector
                image_internal = PostgreImageBaseInternal(
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
                character_name = props["character_name"],
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
                img_path = (self._config.data.agentic_data_root + row["image_path"]).replace("//", "/")
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
                    "character_name": row["character_name"],
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

        return_props = list(PostgreImageBase.model_fields.keys())
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

    def _find_texts_content_relative_lexical_search(
        self,
        query: str,
        top_k: int = 5,
        search_in_character_name: bool = True,
        search_in_body_part: bool = True,
        search_in_file_name: bool = True,
    ) -> list[PostgreText]:
        """
        Perform a lexical search for `AgenticTexts`s using a query string.
        This searches in the character_name, body_part, file_name fields.

        Args:
            query (str): The search query string.
            top_k (int, optional): The number of top results to return. Defaults to 5.
            search_in_character_name (bool, optional): Whether to search in the character_name field. Defaults to True.
            search_in_body_part (bool, optional): Whether to search in the body_part field. Defaults to True.
            search_in_file_name (bool, optional): Whether to search in the file_name field. Defaults to True.

        Returns:
            list[PostgreText]: A list of `AgenticTexts` that match the search query.
        """ 

        collection = self._get_client().collections.get(AGENTIC_TEXTS_SCHEMA_NAME)
        query_properties = []
        if search_in_body_part:
            query_properties.append("body_part")
        if search_in_file_name:
            query_properties.append("file_name")

        if len(query_properties) == 0:
            raise ValueError("At least one property must be selected for lexical search.")
        
        res = collection.query.bm25(
            query=query,
            query_properties=query_properties,
            limit=top_k,
        )

        results = self._create_agentic_texts_from_query_results(res)
        return results
    
    def _find_images_relative_lexical_search(
        self,
        query: str,
        top_k: int = 5,
        search_in_image_name: bool = True,
        search_in_file_name: bool = True,
        search_in_body_part: bool = True,
    ) -> list[PostgreImageBase]:
        """
        Perform a lexical search for `AgenticImages`s using a query string.
        This searches in the image_name, file_name, body_part fields.

        Args:
            query (str): The search query string.
            top_k (int, optional): The number of top results to return. Defaults to 5.
            search_in_image_name (bool, optional): Whether to search in the image_name field. Defaults to True.
            search_in_file_name (bool, optional): Whether to search in the file_name field. Defaults to True.
            search_in_body_part (bool, optional): Whether to search in the body_part field. Defaults to True.

        Returns:
            list[PostgreImageBase]: A list of `AgenticImages` that match the search query.
        """ 

        collection = self._get_client().collections.get(AGENTIC_IMAGES_SCHEMA_NAME)
        query_properties = []
        if search_in_image_name:
            query_properties.append("image_name")
        if search_in_file_name:
            query_properties.append("file_name")
        if search_in_body_part:
            query_properties.append("body_part")

        if len(query_properties) == 0:
            raise ValueError("At least one property must be selected for lexical search.")
        
        
        res = collection.query.bm25(
            query=query,
            query_properties=query_properties,
            return_properties=list(PostgreImageBase.model_fields.keys()),
            limit=int(top_k),
        )

        results = self._create_agentic_images_from_query_results(res)
        return results
    
    def get_image_record_internal_by_vector_id(
            self, 
            vector_id: str,
            include_vector: bool | list[Literal["images_img", "images_description"]] = False,
    ) -> PostgreImageBaseInternal:
        """
        Get a `PostgreImageBaseInternal` by its unique identifier.

        Args:
            vector_id (str): The unique identifier of the record in the VectorDB.

        Returns:
            `PostgreImageBaseInternal`: The `PostgreImageBaseInternal` object with the specified `vector_id`.
        """

        return_props = list(PostgreImageBase.model_fields.keys())
        collection = self._get_client().collections.get(AGENTIC_IMAGES_SCHEMA_NAME)
        res = collection.query.fetch_objects(
            filters=Filter.by_property("vector_id").equal(vector_id),
            return_properties=return_props,
            include_vector=["images_img","images_description"] if include_vector is True else False,
        )
        if len(res.objects) == 0:
            raise KeyError(f"No image record found with vector_id: {vector_id}")

        res = self._create_agentic_images_from_query_results(res)
        if not isinstance(res[0], PostgreImageBaseInternal):
            raise ValueError(f"Image record with vector_id: {vector_id} does not contain vector data.")
        return res[0]
    
        
    @mlflow.trace(
        span_type=SpanType.TOOL,
    )
    def find_agentic_images_with_similar_image(                #m2m
        self,
        vector_id: str,
        top_k: int = 5,
    ) -> list[AgenticImagesSemanticSearchResult]:
        """
        Find `ImageRecord`s with images similar to the one specified by the `vector_id`. This uses semantic similarity search based on image embeddings.

        Args:
            vector_id (str): The unique identifier of the image in the VectorDB.
            top_k (int, optional): Number of top results to return. Defaults to 5

        Returns:
            list[AgenticImagesSemanticSearchResult]: `ImageSimilarity`s search results with similarity scores.
        """
        record = self.get_image_record_internal_by_vector_id(vector_id)
        image_embedding = record.embeddings.get("images_img")
        results = self._agentic_images_img_similarity_search(
            image_embedding,
            top_k=top_k,
        )
        return results
    
    @mlflow.trace(
        span_type=SpanType.TOOL,
    )
    def find_similar_images_img_according_to_users_text_query(                  #t2m
        self,
        query:str,
        top_k: int = 5,
    ) -> list[AgenticImagesSemanticSearchResult]:
        """
        Find `ImageRecord`s with images similar to the text query. This uses semantic similarity search based on image embeddings.

        Args:
            query (str): The text query to find similar images for.
            top_k (int, optional): Number of top results to return. Defaults to 5

        Returns:
            list[AgenticImagesSemanticSearchResult]: `ImageSimilarity`s search results with similarity scores.
        """
        query = self._query_rewriter.rewrite_user_query_for_cross_modal_text_to_image_search(query)
        query_embedding = self._agentic_ml_client.compute_text_embedding(
            query, return_tensor="np"
        ).tolist()  # type: ignore
        results = self._agentic_images_img_similarity_search(
            query_embedding,
            top_k=top_k,
        )
        return results

    @mlflow.trace(
        span_type=SpanType.TOOL,
    )
    def find_similar_images_img_according_to_users_image_query(          #m2m
        self,
        user_image_id: str,
        top_k: int = 5,
    )-> list[AgenticImagesSemanticSearchResult]:
        """
        Find `ImagesRecord`s with images similar to the user-provided image.
        This uses image similarity search based on image embeddings.
        Use this if a user provides an image and you want to find similar images in the database.

        Args:
            user_image_id (str): The unique identifier of the user-provided image.
            top_k (int, optional): Number of top results to return. Defaults to 5

        Returns:
            list[AgenticImagesSemanticSearchResult]: `ImageSimilarity`s search results with similarity scores.
        """
        base64_user_image: str = self._user_image_store.load_user_image(user_image_id, base64=True)  # type: ignore
        base64_user_image = base64_user_image.split(",")[-1]
        image_embedding = self._agentic_ml_client.compute_image_embedding(
            base64_image= base64_user_image, return_tensor="np" 
        ).tolist()  # type: ignore
        results = self._agentic_images_img_similarity_search(
            image_embedding,
            top_k=top_k,
        )
        return results
    
    @mlflow.trace(
        span_type=SpanType.TOOL,
    )
    def find_similar_images_description_according_to_users_text_query(          #t2t
        self,
        query: str,
        top_k: int = 5,
    ) -> list[AgenticImagesSemanticSearchResult]:
        """
        Find `ImageRecord`s with image descriptions similar to the text query.
        The uses textual semantic similarity search based on description embeddings.
        Use this only to search for imagesRecord based on the similarity between the text query and the image descriptions in the database.
        if you want to search for images record based on their images, use `find_agentic_images_with_similar_image` instead.

        Args:
            query (str): The text query to find similar image descriptions for.
            top_k (int, optional): Number of top results to return. Defaults to 5

        Returns:
            list[AgenticImagesSemanticSearchResult]: `ImageSimilarity`s search results with similarity scores.
        """
        # query = self._query_rewriter.rewrite_user_query_for_cross_modal_text_to_image_search(query)
        query_embedding = self._agentic_ml_client.compute_text_embedding(
            query, return_tensor="np"
        ).tolist()  # type: ignore
        results = self._agentic_images_description_similarity_search(
            query_embedding,
            top_k=top_k,
        )
        return results
    
    
    @mlflow.trace(
        span_type=SpanType.TOOL,
    )
    def find_similar_images_description_according_to_users_image_query(
        self,
        user_image_id: str,
        top_k: int = 5,
    ) -> list[AgenticImagesSemanticSearchResult]:
        """
        Find `ImageRecord`s with image descriptions similar to the user-provided image.
        The uses cross-modal semantic similarity search based on description embeddings and image embeddings.
        Use this only to search for imagesRecord based on the similarity between the user-provided image and the image descriptions in the database.
        if you want to search for images record based on their images, use `find_agentic_images_with_similar_image` instead.

        Args:
            user_image_id (str): The unique identifier of the user-provided image.
            top_k (int, optional): Number of top results to return. Defaults to 5

        Returns:
            list[AgenticImagesSemanticSearchResult]: `ImageSimilarity`s search results with similarity scores.
        """
        base64_user_image: str = self._user_image_store.load_user_image(user_image_id, base64=True)  # type: ignore
        base64_user_image = base64_user_image.split(",")[-1]
        image_embedding = self._agentic_ml_client.compute_image_embedding(
            base64_image= base64_user_image, return_tensor="np" 
        ).tolist()  # type: ignore
        results = self._agentic_images_description_similarity_search(
            image_embedding,
            top_k=top_k,
        )
        return results
    
    @mlflow.trace(
        span_type=SpanType.TOOL,
    )
    def find_similar_texts_content_according_to_users_text_query(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[AgenticTextsSemanticSearchResult]:
        """
        Find `TextRecord`s with text content similar to the text query.
        The uses textual semantic similarity search based on text content embeddings.
        Use this only to search for text records based on the similarity between the text query and the text content in the database.

        Args:
            query (str): The text query to find similar text content for.
            top_k (int, optional): Number of top results to return. Defaults to 10

        Returns:
            list[AgenticTextsSemanticSearchResult]: `TextSimilarity`s search results with similarity scores.
        """
        query = self._query_rewriter.rewrite_user_query_for_text_to_text_search(query)
        query_embedding = self._model.encode(query).tolist()
        results = self._agentic_texts_similarity_search(
            query_embedding,
            target_vector="text_content",
            top_k=top_k,
        )
        return results
    
    @mlflow.trace(
        span_type=SpanType.TOOL,
    )
    def find_similar_texts_content_according_to_images_description(
        self,
        vector_id: str,
        top_k: int = 5,
    )-> list[AgenticTextsSemanticSearchResult]:
        """
        Find `TextRecord`s with text content similar to the description of the image with the given vector ID.

        Args:
            vector_id (str): The unique identifier of the image for which to find similar text content.
            top_k (int, optional): Number of top results to return. Defaults to 5.

        Returns:
            list[AgenticTextsSemanticSearchResult]: `TextSimilarity`s search results with similarity scores.
        """
        record = self.get_image_record_internal_by_vector_id(vector_id)
        query_embedding = self._model.encode(record.description).tolist()
        result = self._agentic_texts_similarity_search(
            query_embedding,
            target_vector="text_content",
            top_k=top_k,
        )

        return result

    @mlflow.trace(
        span_type=SpanType.TOOL,
    )
    def find_texts_content_relative_lexical_search(
        self,
        query: str,
        *,
        top_k: int = 5
    ) -> list[PostgreText]:
        """
        Perform a lexical search for `AgenticTexts`s using a query string.
        This searches in the character_name, body_part, file_name fields.

        Args:
            query (str): The search query string.
            top_k (int, optional): The number of top results to return. Defaults to 10.

        Returns:
            list[PostgreText]: A list of `AgenticTexts` that match the search query.
        """
        results = self._find_texts_content_relative_lexical_search(
            query=query,
            top_k=int(top_k),
            search_in_character_name=True,
            search_in_body_part=True,
            search_in_file_name=True,
        )
        return results
    

    @mlflow.trace(
        span_type=SpanType.TOOL,
    )
    def find_images_relative_lexical_search(
        self,
        query: str,
        *,
        top_k: int = 5
    ) -> list[PostgreImageBase]:
        """
        Perform a lexical search for `AgenticImages`s using a query string.
        This searches in the image_name, body_part, file_name fields.

        Args:
            query (str): The search query string.
            top_k (int, optional): The number of top results to return. Defaults to 5.

        Returns:
            list[PostgreImageBase]: A list of `PostgreImageBase` that match the search query.
        """
        results = self._find_images_relative_lexical_search(
            query=query,
            top_k=int(top_k),
            search_in_image_name=True,
            search_in_body_part=True,
            search_in_file_name=True,
        )
        return results
    
    @mlflow.trace(
        span_type=SpanType.TOOL,
    )
    def find_surrounding_chunks(
        self,
        vector_id:str,
        window_size: int = 5,
    ) -> list[PostgreText]:
        """
        Find surrounding text chunks of a given text chunk specified by its vector ID. This is useful for providing more context around a specific text chunk.  

        Args:
            vector_id (str): The unique identifier of the text chunk in the VectorDB.
            window_size (int, optional): The number of surrounding chunks to retrieve on each side. Defaults to 5.
        Returns:
            list[PostgreText]: A list of `PostgreText` records representing the surrounding text chunks, ordered from the farthest previous chunk to the farthest next chunk.
        """
        results = self._find_surrounding_chunks(
            vector_id=vector_id,
            window_size=window_size,
        )
        return results
    
    def _find_surrounding_chunks(
        self,
        vector_id:str,
        window_size: int = 5,
    ) -> list[PostgreText]:
        """
        Find surrounding text chunks of a given text chunk specified by its vector ID. This is useful for providing more context around a specific text chunk.  

        Args:
            vector_id (str): The unique identifier of the text chunk in the VectorDB.
            window_size (int, optional): The number of surrounding chunks to retrieve on each side. Defaults to 5. 

        Returns:
            list[PostgreText]: A list of `PostgreText` records representing the surrounding text chunks, ordered from the farthest previous chunk to the farthest next chunk.
        """
        collection = self._get_client().collections.get(AGENTIC_TEXTS_SCHEMA_NAME)
        target_chunk = collection.query.fetch_objects(
            filters=Filter.by_property("vector_id").equal(vector_id),
            limit=1,
        )
        if target_chunk is None or len(target_chunk.objects) == 0:
            raise ValueError(f"No text chunk found with vector ID: {vector_id}")
        
        target_chunk_props = target_chunk.objects[0].properties
        chunk_index = target_chunk_props.get("chunk_index")
        doc_id = target_chunk_props.get("doc_id")
        if chunk_index is None or doc_id is None:
            raise ValueError(
                f"Missing chunk metadata for vector ID: {vector_id}. "
                "Expected 'chunk_index' and 'doc_id'."
            )

        filter = Filter.all_of(
            [
                Filter.by_property("doc_id").equal(doc_id),
                Filter.by_property("chunk_index").greater_or_equal(chunk_index - window_size),
                Filter.by_property("chunk_index").less_or_equal(chunk_index + window_size),
            ]
        )

        res = collection.query.fetch_objects(
            filters=filter,
            return_properties=list(PostgreText.model_fields.keys()),
            limit= 2*window_size + 1,  # the target chunk itself + surrounding chunks on both sides
        )

        results = self._create_agentic_texts_from_query_results(res)

        # sort the results by chunk_index
        results.sort(key=lambda x: x.chunk_index)

        return results

    @mlflow.trace(
        span_type=SpanType.TOOL,
    )
    def get_text_record_by_vector_id(
        self,
        vector_id: str,
    ) -> PostgreText:
        """
        Retrieve the text record of a text chunk specified by its vector ID.

        Args:
            vector_id (str): The unique identifier of the text chunk in the VectorDB.

        Returns:
            PostgreText: The text record of the specified text chunk.
        """
        collection = self._get_client().collections.get(AGENTIC_TEXTS_SCHEMA_NAME)
        target_chunk = collection.query.fetch_objects(
            filters=Filter.by_property("vector_id").equal(vector_id),
            limit=1,
        )
        if len(target_chunk.objects) == 0:
            raise ValueError(f"No text chunk found with vector ID: {vector_id}")

        results = self._create_agentic_texts_from_query_results(target_chunk)
        
        return results[0]

    @mlflow.trace(
        span_type=SpanType.TOOL,
    )
    def get_image_record_by_vector_id(
        self,
        vector_id: str,
    ) -> PostgreImageBase:
        """
        Retrieve the image record of an image specified by its vector ID.

        Args:
            vector_id (str): The unique identifier of the image in the VectorDB.

        Returns:
            PostgreImageBase: The image record of the specified image.
        """
        collection = self._get_client().collections.get(AGENTIC_IMAGES_SCHEMA_NAME)
        res = collection.query.fetch_objects(
            filters=Filter.by_property("vector_id").equal(vector_id),
            return_properties=list(PostgreImageBase.model_fields.keys()),
            limit=1,
        )
        if len(res.objects) == 0:
            raise ValueError(f"No image found with vector ID: {vector_id}")

        results = self._create_agentic_images_from_query_results(res)
        
        return results[0]
    
    def get_base_image_by_vector_id(
        self,
        vector_id: str,
    ) -> PostgreImage:
        """
        Retrieve the image record of an image specified by its vector ID.

        Args:
            vector_id (str): The unique identifier of the image in the VectorDB.

        Returns:
            PostgreImage: The image record of the specified image.
        """
        collection = self._get_client().collections.get(AGENTIC_IMAGES_SCHEMA_NAME)
        res = collection.query.fetch_objects(
            filters=Filter.by_property("vector_id").equal(vector_id),
            return_properties=list(PostgreImage.model_fields.keys()),
            limit=1,
        )
        if len(res.objects) == 0:
            raise ValueError(f"No image found with vector ID: {vector_id}")

        res_obj = res.objects[0]
        props = res_obj.properties

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
            base64_image=props["base64_image"]
        )
        
        return image_record

    @mlflow.trace(
        span_type=SpanType.TOOL,
    )
    def list_all_body_parts(self) -> list[str]:
        """
        List all unique body parts mentioned in the database. This can be useful for understanding the coverage of the data and for providing options to users when they want to filter or search by body part.

        Returns:
            list[str]: A list of unique body parts mentioned in the database.
        """
        text_body_parts = self._texts_df["body_part"].unique().tolist()
        image_body_parts = self._images_df["body_part"].unique().tolist()
        body_parts = set(text_body_parts + image_body_parts)

        return body_parts
    
    @mlflow.trace(
        span_type=SpanType.TOOL,
    )
    def list_all_file_names(self) -> list[str]:
        """
        List all unique file names mentioned in the database. This can be useful for understanding the coverage of the data and for providing options to users when they want to filter or search by file name.

        Returns:
            list[str]: A list of unique file names mentioned in the database.
        """
        text_file_names = self._texts_df["file_name"].unique().tolist()
        file_names = set(text_file_names)

        return file_names

    @mlflow.trace(
        span_type=SpanType.TOOL,
    )
    def list_all_character_names_in_one_file(
        self,
        file_name: str,
    ) -> list[str]:
        """
        List all unique character names mentioned in a specific file. This can be useful for understanding the coverage of the data in that file and for providing options to users when they want to filter or search by character name within that file.

        Args:
            file_name (str): The name of the file for which to list character names.

        Returns:
            list[str]: A list of unique character names mentioned in the specified file.
        """
        characters_in_file = self._texts_df[self._texts_df["file_name"] == file_name]["character_name"].unique().tolist()
        return characters_in_file