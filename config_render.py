import os
from pydantic_settings import BaseSettings

class SettingsRender(BaseSettings):
    # API Keys
    openai_api_key: str = "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"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-3.5-turbo"  # Faster for Render
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./hackrx_db.db")
    
    # Render-optimized processing settings
    max_chunk_size: int = 400    # Smaller for speed
    chunk_overlap: int = 30      # Minimal overlap
    max_tokens: int = 50         # Very short responses
    max_context_chunks: int = 2  # Limit context for speed
    
    # Security
    bearer_token: str = "433c9562217435ac71d779508405bfa9b20d0f58ff2aeb482c16c0e251f9f85f"
    
    # Render-specific settings
    environment: str = "render"
    timeout_multiplier: float = 0.8  # More aggressive timeouts
    
    class Config:
        env_file = ".env"

settings = SettingsRender()