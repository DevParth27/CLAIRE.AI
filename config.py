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
# Dynamic Document Parsing RAG Configuration - Maximum Accuracy

# Chunking settings - optimized for document structure preservation
    max_chunk_size: int = 1024             # Optimal for preserving document context
    chunk_overlap: int = 256               # 25% overlap for document continuity
    max_tokens: int = 32000               # Maximum context for complete document understanding
    max_context_chunks: int = 100         # Very high for variable document complexity
    batch_size: int = 32                  # Dynamic batch processing capability
    parallel_questions: int = 16          # High parallelization for variable question loads

# Vector search settings - maximum precision for document parsing
    vector_top_k: int = 200               # High retrieval for comprehensive coverage
    similarity_threshold: float = 0.005   # Very low threshold to capture all relevant content
    
    # Security
    bearer_token: str = os.getenv("BEARER_TOKEN", "")
    
    # Production settings
    debug: bool = False
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()