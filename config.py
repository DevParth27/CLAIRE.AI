import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Keys - Switched to Gemini
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./hackrx_db.db")
    
    # Enhanced processing settings for Gemini-2.5-flash
    # Dynamic Document Parsing RAG Configuration - Maximum Accuracy

    # Chunking settings - optimized for Gemini-2.5-flash context window
    max_chunk_size: int = 1536             # Increased for better context preservation
    chunk_overlap: int = 384               # 25% overlap maintained for continuity
    max_tokens: int = 128000              # Increased for Gemini-2.5-flash's larger context window
    max_context_chunks: int = 120         # Increased for more comprehensive document analysis
    batch_size: int = 48                  # Increased for better throughput
    parallel_questions: int = 24          # Increased parallelization for better performance

    # Vector search settings - precision-optimized for Gemini-2.5-flash
    vector_top_k: int = 250               # Increased retrieval for comprehensive coverage
    similarity_threshold: float = 0.003   # Lower threshold to capture more relevant content
    
    # Gemini-2.5-flash specific settings
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