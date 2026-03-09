from weaviate.classes.config import (
    Configure,
    DataType,
    Property,
    ReferenceProperty,
    Tokenization,
    VectorDistances,
)

AGENTIC_IMAGES_SCHEMA_NAME = "AgenticImages"
AGENTIC_IMAGES_SCHEMA_PROPS = [
    Property(
        name="postgre_id",
        data_type=DataType.UUID,
        description="Unique identifier for the image record, matching the ID in the PostgreDB.",
        skip_vectorization=True,
        vectorize_property_name=False,
    ),
    Property(
        name="vector_id",
        data_type=DataType.UUID,
        description="Unique identifier for the image embedding vector, matching the ID in the PostgreDB embeddings table.",
        skip_vectorization=True,
        vectorize_property_name=False,
    ),
    Property(
        name= "image_name",
        data_type=DataType.TEXT,
        tokenization=Tokenization.WORD,
        description="The name of the image file.",
        skip_vectorization=True,
        vectorize_property_name=False,
    ),
    Property(
        name= "page_number",
        data_type=DataType.INT,
        description="The page number in the original PDF document where the image is located.",
        skip_vectorization=True,
        vectorize_property_name=False,  
    ),
    Property(
        name= "image_index",
        data_type=DataType.INT,
        description="The index of the image within the original PDF document.",
        skip_vectorization=True,
        vectorize_property_name=False,
    ),
    Property(
        name = "image_path",
        data_type=DataType.TEXT,
        tokenization=Tokenization.FIELD,
        description="The file path to the image on disk.",
        skip_vectorization=True,
        vectorize_property_name=False,
    ),
    Property(
        name= "file_name",
        data_type=DataType.TEXT,
        tokenization=Tokenization.FIELD,
        description="The name of the original PDF file from which the image was extracted.",
        skip_vectorization=True,
        vectorize_property_name=False,
    ),
    Property(
        name= "description",
        data_type=DataType.TEXT,
        tokenization=Tokenization.WORD,
        description="A textual description of the image content, if available.",
        skip_vectorization=True,
        vectorize_property_name=False,
    ),
    Property(
        name= "body_part",
        data_type=DataType.TEXT,
        tokenization=Tokenization.FIELD,
        description="The body part depicted in the image, if known.",
        skip_vectorization=True,
        vectorize_property_name=False,
    ),
    Property(
        name= "base64_image",
        data_type=DataType.BLOB,
        description="The image data encoded in base64 format.",
        skip_vectorization=True,
        vectorize_property_name=False,
    )
]
AGENTIC_IMAGES_SCHEMA_VECTORIZER = [
    Configure.NamedVectors.none(
        name="images_img",
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE,
        ),
    ),
    Configure.NamedVectors.none(
        name="images_description",
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE
        ),
    ),
]

AGENTIC_TEXTS_SCHEMA_NAME = "AgenticTexts"
AGENTIC_TEXTS_SCHEMA_PROPS = [
    Property(
        name="doc_id",
        data_type=DataType.UUID,
        description="Unique identifier for the text record, matching the ID in the PostgreDB.",
        skip_vectorization=True,
        vectorize_property_name=False,
    ),
    Property(
        name="vector_id",
        data_type=DataType.UUID,
        description="Unique identifier for the text embedding vector, matching the ID in the PostgreDB embeddings table.",
        skip_vectorization=True,
        vectorize_property_name=False,
    ),
    Property(
        name = "character_path",
        data_type=DataType.TEXT,
        tokenization=Tokenization.FIELD,
        description="The file path to the character text file on disk.",
        skip_vectorization=True,
        vectorize_property_name=False,
    ),
    Property(
        name= "file_name",
        data_type=DataType.TEXT,
        tokenization=Tokenization.FIELD,
        description="The name of the original PDF file from which the text was extracted.",
        skip_vectorization=True,
        vectorize_property_name=False,
    ),
    Property(
        name= "body_part",
        data_type=DataType.TEXT,
        tokenization=Tokenization.FIELD,
        description="The body part depicted in the image, if known.",
        skip_vectorization=True,
        vectorize_property_name=False,
    ),
    Property(
        name = "content",
        data_type=DataType.TEXT,
        tokenization=Tokenization.WORD,
        description="The textual content of the document chunk.",
    ),
    Property(
        name= "chunk_index",
        data_type=DataType.INT,
        description="The index of the text chunk within the original document.",
        skip_vectorization=True,
        vectorize_property_name=False,  
    )
]
AGENTIC_TEXTS_SCHEMA_VECTORIZER = [
    Configure.NamedVectors.none(
        name="text_content",
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE,
        ),
    ),
]
