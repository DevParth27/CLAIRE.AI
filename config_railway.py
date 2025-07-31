import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Keys - Railway environment variables
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-3.5-turbo"
    
    # Pinecone (Optional)
    pinecone_api_key: Optional[str] = os.getenv("PINECONE_API_KEY")
    pinecone_environment: Optional[str] = os.getenv("PINECONE_ENVIRONMENT")
    pinecone_index_name: Optional[str] = os.getenv("PINECONE_INDEX_NAME")
    
    # Database - Railway provides DATABASE_URL
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./hackrx_db.db")
    
    # Railway-optimized settings (smaller for free tier)
    max_chunk_size: int = 300    # Smaller chunks for Railway
    chunk_overlap: int = 30      # Minimal overlap
    max_tokens: int = 80         # Shorter responses
    max_context_chunks: int = 2  # Limit context for Railway
    
    # Security
    bearer_token: str = os.getenv("BEARER_TOKEN", "")
    
    # Railway specific
    port: int = int(os.getenv("PORT", 8000))
    railway_environment: str = os.getenv("RAILWAY_ENVIRONMENT", "production")
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()