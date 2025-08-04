import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Keys - Switched to Gemini
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./hackrx_db.db")
    
    # Enhanced processing settings for accuracy
    max_chunk_size: int = 2500             # Increased for better context
    chunk_overlap: int = 500               # Better overlap for continuity
    max_tokens: int = 4000                 # Increased for detailed answers
    max_context_chunks: int = 15           # More context for accuracy
    batch_size: int = 10                   # Support for batch processing
    parallel_questions: int = 5            # Number of questions to process in parallel
    
    # Vector search settings
    vector_top_k: int = 12                 # Increased for better context retrieval
    similarity_threshold: float = 0.15     # Lower threshold for more comprehensive retrieval
    
    # Security
    bearer_token: str = os.getenv("BEARER_TOKEN", "")
    
    # Production settings
    debug: bool = False
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()