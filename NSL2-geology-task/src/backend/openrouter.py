import os
from contextlib import contextmanager
from typing import Any

from loguru import logger
from openai import OpenAI

from src.genner import get_genner
from src.typing.config import AppConfig

from .session import BackendSession


@contextmanager
def setup_openrouter(app_config: AppConfig) -> Any:
    """Set up OpenRouter backend using OpenAI client."""
    
    # Get API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is required for OpenRouter backend")
    
    # Extract model name (remove "openrouter:" prefix)
    if app_config.model_name.startswith("openrouter:"):
        model_name = app_config.model_name.split(":", 1)[1].strip()
    else:
        model_name = app_config.model_name
    
    # OpenRouter API endpoint
    base_url = "https://openrouter.ai/api/v1"
    
    logger.info(f"Setting up OpenRouter client for model: {model_name}")
    logger.info(f"OpenRouter base URL: {base_url}")
    logger.info(f"API key present: {bool(api_key)}")
    
    # Create OpenAI client configured for OpenRouter
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=app_config.inference.timeout,
        max_retries=3,
    )
    
    # Create a config object that matches ServerConfig interface
    class OpenRouterConfig:
        def __init__(self):
            self.name = "openrouter"
            self.model = model_name
            self.endpoint = base_url
            self.api_key = api_key
            self.timeout = app_config.inference.timeout
            self.max_tokens = getattr(app_config.inference, 'max_tokens', None)
            self.temperature = 0.7  # Default temperature
            self.top_p = 0.9  # Default top_p
            self.frequency_penalty = None
            self.presence_penalty = None
            self.enable_auto_tool_choice = True
            self.tool_call_parser = None  # OpenRouter handles this automatically
    
    config = OpenRouterConfig()
    
    # Get the genner (using vllm genner since it works with OpenAI-compatible APIs)
    genner = get_genner("vllm", server_config=config, oai_client=client)
    
    # Create backend session
    session = BackendSession(
        genner=genner,
        smoke_test=lambda: "OpenRouter connection ready",
        client=client,
        config=config,
        base_url=base_url,
        models_url=f"{base_url}/models",
        process=None,
        metrics_url=None,
    )
    
    yield session
