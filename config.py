import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Keys - Switched to Gemini
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./hackrx_db.db")
    
    # Enhanced processing settings for Gemini-2.5-pro
    # Dynamic Document Parsing RAG Configuration - Maximum Accuracy

    # Chunking settings - optimized for Gemini-2.5-pro context window
# Optimized for Gemini-2.5-Pro
    max_chunk_size: int = 1024
    chunk_overlap: int = 256
    max_tokens: int = 65536  # Matches Pro's max output tokens
    max_context_chunks: int = 120
    batch_size: int = 40  # Safer for hackathon infra
    parallel_questions: int = 12

    # Retrieval
    vector_top_k: int = 250
    similarity_threshold: float = 0.005  # Better balance of recall/precision

# Generation
    temperature: float = 0.01
    top_p: float = 0.95
    top_k: int = 40
                   # Adjusted for better token selection
    
    # Security
    bearer_token: str = os.getenv("BEARER_TOKEN", "")
    
    # Production settings
    debug: bool = False
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()