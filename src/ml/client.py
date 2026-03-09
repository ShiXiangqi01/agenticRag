import numpy as np
from src.ml.dto import EmbeddingsInput, EmbeddingsOutput
from src.singleton_meta import SingletonMeta
from typing import TYPE_CHECKING, Literal
from src.config import load_config
from loguru import logger
import requests
import time
if TYPE_CHECKING:
    import torch

class AgenticMLClient(metaclass=SingletonMeta):
    def __init__(self, agentic_ml_url: str | None = None):
        if agentic_ml_url is not None:
            self.agentic_ml_url = agentic_ml_url
        else:
            config = load_config()
            self.agentic_ml_url = config.agentic.ml_url

        self.__wait_for_ready()

    def __wait_for_ready(self, s:int = 60, sleep_t: int = 3) -> None:
        while s  > 0:
            if self.is_ready():
                logger.info("Agentic ML 服务已准备就绪")
                return
            logger.info("等待 Agentic ML 服务准备就绪...")
            time.sleep(sleep_t)
            s -= sleep_t
        
        raise TimeoutError("等待 Agentic ML 服务准备就绪超时")
    
    def is_ready(self) -> bool:
        try:
            return requests.get(f"{self.agentic_ml_url}/health").status_code == 200
        except Exception:
            return False
        
    def compute_image_embedding(
            self,
            base64_image:str,
            return_tensor: Literal["pt" , "np"] | None = "np",
    )-> "EmbeddingsOutput | np.ndarray | torch.Tensor":
        """
        Get the embedding for a base64 encoded image.
        """

        input = EmbeddingsInput(input_data=base64_image, input_type="image")
        return self._get_embeddings(input, return_tensor, squeeze=True)
        
    def compute_text_embedding(
            self,
            text: str,
            return_tensor: Literal["pt" , "np"] | None = "np",
    ) -> "EmbeddingsOutput | np.ndarray | torch.Tensor":
        """
        Get the embedding for a text input.
        """
        input = EmbeddingsInput(input_data=text, input_type="text")
        return self._get_embeddings(input, return_tensor, squeeze=True)
    
    def _get_embeddings(
        self,
        input: EmbeddingsInput,
        return_tensor: Literal["pt" , "np"] | None = "np",
        squeeze: bool = True,
    ) -> "EmbeddingsOutput | np.ndarray | torch.Tensor":
        response = requests.post(f"{self.agentic_ml_url}/embeddings", json=input.model_dump())
        response.raise_for_status()
        response_json = response.json()
        emb = EmbeddingsOutput.model_validate(response_json)  #如果数据不符合模型定义（如缺少必填字段、类型错误），会抛出 ValidationError

        if return_tensor == "pt":
            import torch

            emb = torch.tensor(emb.embeddings)
            if squeeze:
                emb = emb.squeeze()
        elif return_tensor == "np":
            emb = np.array(emb.embeddings)
            if squeeze:
                emb = emb.squeeze()
        else:
            if squeeze and len(emb.embeddings) == 1 and isinstance(emb.embeddings[0], list):
                emb.embeddings = emb.embeddings[0]
        return emb