import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Keys - Use environment variables instead of hardcoded values
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-3.5-turbo"
    
    # Pinecone (Optional since you might not always use it)
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_index_name: Optional[str] = None
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./hackrx_db.db")
    
    # Memory-optimized processing settings
    max_chunk_size: int = 500
    chunk_overlap: int = 50
    max_tokens: int = 100
    max_context_chunks: int = 3
    
    # Security - Use environment variable instead of hardcoded
    bearer_token: str = os.getenv("BEARER_TOKEN", "")
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()