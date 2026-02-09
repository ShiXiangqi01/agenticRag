import base64
import io
import os
from fastapi import logger
import litserve as ls

from loguru import logger
from transformers import SiglipModel, SiglipProcessor, BatchFeature
import torch

from ml.dto import EmbeddingsInput, EmbeddingsOutput
from PIL import Image

MODEL_NAME = "google/siglip-so400m-pathch14-384"
TORCH_DTYPE = torch.bfloat16
AGENTIC_ML_DEV_MODE = int(os.environ.get("AGENTIC_ML_DEV_MODE", "1"))

class SigLipLitAPI(ls.LitAPI):
    def setup(self, device:str):
        self.device: str = device
        logger.info(f"Using device: {device}")
        self.model = SiglipModel.from_pretrained(
            MODEL_NAME,
            attn_implementation="flash_attention_2",
            torch_dtype=TORCH_DTYPE,
            device_map=device

        )
        self.processor = SiglipProcessor.from_pretrained(MODEL_NAME)

    def _compute_text_embeddings(self, text_features:BatchFeature) -> torch.Tensor:
        with torch.no_grad():
            text_emb = self.model.get_text_features(**text_features.to(self.device))
        return text_emb

    def _compute_image_embeddings(self, image_features:BatchFeature) -> torch.Tensor:
        with torch.no_grad():
            img_emb = self.model.get_image_features(**image_features.to(self.device, dtype=TORCH_DTYPE))
        return img_emb


    def decode_request(self, request: EmbeddingsInput) -> dict[str, list | None]:
        if request.input_type == "text":
            text = request.input_data if isinstance(request.input_data, list) else [request.input_data]
            image = None
        elif request.input_type == "image":
            image_data = request.input_data if isinstance(request.input_data, list) else [request.input_data]
            image = [Image.open(io.BytesIO(base64.b64decode(b64))) for b64 in image_data]
            text = None
        else:
            raise ValueError("Invalid input_type. Must be 'text' or 'image'.")
        
        return {"image": image, "text": text}
    
    def predict(self, inputs: dict[str, list | None]) -> torch.Tensor:
        images = inputs["image"]
        texts = inputs["text"]

        if images is None and texts is None:
            raise ValueError("At least one of 'image' or 'text' must be provided.")
        elif images is not None and texts is not None:
            raise ValueError("Only one of 'image' or 'text' should be provided at a time.")
        elif images is not None and len(images) > 0:
            image_features = self.processor(
                images=images,
                padding="max_length",
                return_tensors="pt"
            )
            embs = self._compute_image_embeddings(image_features)
        elif texts is not None and len(texts) > 0:
            text_features = self.processor(
                text=texts,
                padding="max_length",
                truncation=True, #如果长度超过模型允许的最大长度，就自动截断到最大长度，避免超长报错。
                return_tensors="pt"
            )
            embs = self._compute_text_embeddings(text_features)
        else:
            raise ValueError("No valid inputs provided.")
        return embs
    
    def encode_response(self, outputs: torch.Tensor) -> EmbeddingsOutput:
        return EmbeddingsOutput(
            embeddings = outputs.tolist(),
            embeddings_model = MODEL_NAME,
        )
    
if __name__ == "__main__":
    if AGENTIC_ML_DEV_MODE == 0:
        workers_per_device = 2
    else:
        workers_per_device = 1
    
    port = 8000
    logger.info(f"Starting server on port {port}")
    logger.info(f"Workers per device: {workers_per_device}")

    api = SigLipLitAPI()
    server = ls.LitServer(
        api,
        accelerator="cuda",
        api_path="/embed",
        workers_per_device=workers_per_device,
    )
    server.run(port=port)
        

        
        