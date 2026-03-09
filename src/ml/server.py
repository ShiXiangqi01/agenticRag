import base64
import io
import os

import litserve as ls
import torch
from loguru import logger
from PIL import Image
from transformers import BatchFeature, SiglipModel, SiglipImageProcessor, SiglipTokenizer
from dto import EmbeddingsInput, EmbeddingsOutput


MODEL_NAME = "google/siglip-base-patch16-224"
TORCH_DTYPE = torch.bfloat16
AGENTIC_ML_DEV_MODE = int(os.environ.get("AGENTIC_ML_DEV_MODE", "1"))

class SigLipLitAPI(ls.LitAPI):
    def setup(self, device:str):
        self.device: str = device
        logger.info(f"Using device: {device}")
        self.model = SiglipModel.from_pretrained(
            MODEL_NAME,
            torch_dtype=TORCH_DTYPE,
            device_map=device

        )
        self.processor = SiglipImageProcessor.from_pretrained(MODEL_NAME)
        self.tokenizer = SiglipTokenizer.from_pretrained(MODEL_NAME)

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
            image = []
            for i, b64 in enumerate(image_data):
                try:
                    img_bytes = base64.b64decode(b64)
                    img = Image.open(io.BytesIO(img_bytes))
                    img.load()  # Ensure the image is fully loaded
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    image.append(img)
                except Exception as e:
                    logger.error(f"Failed to decode image at index {i}: {e}")
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
                return_tensors="pt"
            )
            embs = self._compute_image_embeddings(image_features)
        elif texts is not None and len(texts) > 0:
            text_features = self.tokenizer(
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
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embeddings_tensor = outputs.pooler_output
        else:

            embeddings_tensor = outputs.last_hidden_state[:, 0, :]
        embeddings_list = embeddings_tensor.detach().cpu().tolist()

        return EmbeddingsOutput(
            embeddings=embeddings_list,
            embedding_model=MODEL_NAME,
        )
    
if __name__ == "__main__":
    if AGENTIC_ML_DEV_MODE == 0:
        workers_per_device = 2
    else:
        workers_per_device = 1
    
    port = 8001
    logger.info(f"Starting server on port {port}")
    logger.info(f"Workers per device: {workers_per_device}")

    api = SigLipLitAPI(api_path="/embeddings")
    server = ls.LitServer(
        api,
        accelerator="cuda",
        workers_per_device=workers_per_device,
    )
    server.run(port=port)
        

        
        