import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Keys
    openai_api_key: str = "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"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-3.5-turbo"
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./hackrx_db.db")
    
    # Memory-optimized processing settings
    max_chunk_size: int = 500    # Smaller chunks
    chunk_overlap: int = 50      # Minimal overlap
    max_tokens: int = 100        # Shorter responses
    max_context_chunks: int = 3  # Limit context
    
    # Security
    bearer_token: str = "433c9562217435ac71d779508405bfa9b20d0f58ff2aeb482c16c0e251f9f85f"
    
    class Config:
        env_file = ".env"

settings = Settings()