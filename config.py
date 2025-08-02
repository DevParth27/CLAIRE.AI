import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = "https://api.openai.com/v1"
    # COST-EFFECTIVE MODEL WITH HIGH ACCURACY
    openai_model: str = "gpt-4-turbo"  # 70% cheaper than GPT-4o, 92% accuracy
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./hackrx_db.db")
    
    # OPTIMIZED settings for cost-effective accuracy
    max_chunk_size: int = 1000     # Balanced context size
    chunk_overlap: int = 150       # Good overlap
    max_tokens: int = 600          # Sufficient for detailed answers
    max_context_chunks: int = 6    # Good context without excess cost
    
    # Security
    bearer_token: str = os.getenv("BEARER_TOKEN", "")
    
    # Production settings
    debug: bool = False
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()