import io
from uuid import uuid4

from loguru import logger
from PIL import Image
from matplotlib import image
from src.singleton_meta import SingletonMeta
from src.config import load_config
from pathlib import Path
from src.data.utils import encode_image

class UserImageStore(metaclass=SingletonMeta):
    def __init__(self):
        config = load_config()
        self._images_root = Path(config.data.user_images_dir)
        self._images_root.mkdir(parents=True, exist_ok=True)
        self._supported_image_formats = {".jpg", ".jpeg", ".png"}
        self._user_images: dict[str, Path] = self._read_user_image()

    def _read_user_image(self):
        user_images = {}
        for ext in self._supported_image_formats:
            for image_path in self._images_root.glob(f"*{ext}"):
                user_images[image_path.stem] = image_path
        logger.info(f"Read {len(user_images)} user images from {self._images_root}")
        return user_images
        

    def store_user_image(self, image_bytes: bytes, mime:str | None) -> str:
        uuid = str(uuid4())
        if mime not in ["image/jpg","image/jpeg", "image/png"]:
            raise ValueError(f"Invalid image format! Supported formats are: jpg, jpeg, png")
        ext = mime.split("/")[-1]
        ofn = self._images_root / f"{uuid}.{ext}"
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image_to_save = image
            if ext in {"jpg", "jpeg", "png"} and image.mode in {"RGBA", "LA", "P"}:
                image_to_save = image.convert("RGB")

            image_to_save.save(ofn)

            self._user_images[uuid] = ofn
            logger.info(f"Stored user image with UUID: {uuid} at {ofn}")
            return uuid
        except Exception as e:
            logger.error(f"Failed to store user image: {e}")
            raise


    

        

    def load_user_image(self, image_id:str, base64: bool = False) -> Image.Image | str:
        if image_id not in self._user_images:
            raise ValueError(f"Image with ID {image_id} not found.")
        image_path = self._user_images[image_id]
        image = Image.open(image_path)
        if base64:
            base64_data = encode_image(image)
            image = f"data:image/jpeg;base64,{base64_data}"
        return image