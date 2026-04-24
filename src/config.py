import os
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pathlib import Path
from loguru import logger

class DataConfig(BaseSettings):
    images_df_file:str
    images_embedding_df_file:str
    texts_df_file:str
    texts_embedding_df_file:str
    agentic_data_root:str
    user_images_dir:str


class AgenticConfig(BaseSettings):
    ml_url: str
    model_name: str

class WeaviateConfig(BaseSettings):
    host:str
    http_port:int
    grpc_port:int
    
class AppConfig(BaseSettings):
    dev_mode: bool
    reset_vdb_on_startup: bool

class AssistantConfig(BaseSettings):
    default_model: str

class Config(BaseSettings):
    
    model_config = SettingsConfigDict(
        env_prefix = "AGENTIC_",
        env_nested_delimiter="__",
        extra="ignore"
    )

    data: DataConfig
    agentic: AgenticConfig
    weaviate: WeaviateConfig
    app: AppConfig
    assistant: AssistantConfig
    
@lru_cache(maxsize=2)
def load_config(config_file: str | Path | None = None) -> Config:
    if config_file is None:
        config_file = os.getenv("AGENTIC_CONFIG_FILE", None)
    if config_file is None:
        config_file = "/home/xiangqi/xiangqi/agenticRag/config/config.dev.yaml"
    
    config_file = Path(config_file)

    if not config_file.exists():
        raise ValueError(f"配置文件不存在: {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)

            cfg = Config.model_validate(config_dict)
            logger.info(f"配置加载成功: {config_file}")
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        raise e
    
    # google_application_credentials_file = Path(cfg.google.application_credentials_file)
    # if not google_application_credentials_file.exists():
    #     raise ValueError(f"Google application credentials file not found at {google_application_credentials_file}")
    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(google_application_credentials_file)
    # os.environ["OPENAI_API_KEY"] = cfg.openai.api_key

    # Log the configuration without sensitive information
    log_config = config_dict.copy()
    # if "openai" in log_config and "api_key" in log_config["openai"]:
    #     log_config["openai"]["api_key"] = "***"
    # logger.info(f"Configuration: {log_config}")

    return cfg
