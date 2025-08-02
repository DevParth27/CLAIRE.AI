import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = "https://api.openai.com/v1"
    # COST-OPTIMIZED MODEL FOR HIGH ACCURACY
    openai_model: str = "gpt-4-turbo"  # Best cost/accuracy ratio
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./hackrx_db.db")
    
    # COST-OPTIMIZED SETTINGS
    max_chunk_size: int = 1200     # Good context without excess cost
    chunk_overlap: int = 200       # Balanced overlap
    max_tokens: int = 600          # Detailed but cost-effective answers
    max_context_chunks: int = 8    # Optimal context for accuracy
    
    # Security
    bearer_token: str = os.getenv("BEARER_TOKEN", "")
    
    # Production settings
    debug: bool = False
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()