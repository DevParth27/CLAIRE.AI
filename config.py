import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Keys - Switched to Gemini
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")  # Changed to flash-lite
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./hackrx_db.db")
    
    # Enhanced processing settings for Gemini-2.5-flash-lite
    # Dynamic Document Parsing RAG Configuration - Maximum Accuracy

    # Chunking settings - optimized for Gemini-2.5-flash-lite context window
    # Reduce token usage by adjusting these parameters
    max_chunk_size: int = 768             # Reduced from 1024
    chunk_overlap: int = 128              # Reduced from 256
    max_tokens: int = 64000               # Reduced from 128000
    max_context_chunks: int = 60          # Reduced from 120
    batch_size: int = 24                  # Reduced from 48
    parallel_questions: int = 6           # Reduced from 12
    
    # Vector search settings
    vector_top_k: int = 100               # Reduced from 250
    similarity_threshold: float = 0.003   # Lower threshold to capture more relevant content
    
    # Gemini-2.5-flash-lite specific settings
    temperature: float = 0.01             # Very low temperature for factual responses
    top_p: float = 0.97                   # Slightly increased for better coverage
    top_k: int = 60                       # Increased for better token selection
    
    # Security
    bearer_token: str = os.getenv("BEARER_TOKEN", "")
    
    # Production settings
    debug: bool = False
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()