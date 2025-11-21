import os
import configparser
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

CONFIG_FILE_NAME = "api_config.ini"
ENV_PREFIX = "HUNYUAN_"

def get_api_config():
    """
    Retrieves API configuration from environment variables or config file.
    Priority:
    1. Environment Variables (HUNYUAN_API_KEY, HUNYUAN_API_URL, HUNYUAN_MODEL_NAME)
    2. api_config.ini in the custom node directory
    3. Defaults
    """
    
    # Defaults
    config = {
        "api_key": "",
        "api_url": "https://api.deepseek.com/v1/chat/completions",
        "model_name": "deepseek-chat"
    }

    # 1. Load from config file
    current_dir = Path(__file__).parent
    config_path = current_dir / CONFIG_FILE_NAME
    
    if config_path.exists():
        try:
            parser = configparser.ConfigParser()
            parser.read(config_path)
            if "API" in parser:
                if "api_key" in parser["API"]:
                    config["api_key"] = parser["API"]["api_key"]
                if "api_url" in parser["API"]:
                    config["api_url"] = parser["API"]["api_url"]
                if "model_name" in parser["API"]:
                    config["model_name"] = parser["API"]["model_name"]
        except Exception as e:
            logger.warning(f"Failed to read {CONFIG_FILE_NAME}: {e}")

    # 2. Override with Environment Variables
    env_key = os.environ.get(f"{ENV_PREFIX}API_KEY")
    env_url = os.environ.get(f"{ENV_PREFIX}API_URL")
    env_model = os.environ.get(f"{ENV_PREFIX}MODEL_NAME")

    if env_key:
        config["api_key"] = env_key
    if env_url:
        config["api_url"] = env_url
    if env_model:
        config["model_name"] = env_model

    return config
