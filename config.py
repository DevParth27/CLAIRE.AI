import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Keys - Switched to Gemini
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./hackrx_db.db")
    
    # Enhanced processing settings for Gemini-1.5-flash
    # Dynamic Document Parsing RAG Configuration - Maximum Accuracy

    # Chunking settings - optimized for Gemini-1.5-flash context window
    # Optimized for Gemini-1.5-flash
    max_chunk_size: int = 512  # Reduced for Flash model
    chunk_overlap: int = 128   # Reduced for Flash model
    max_tokens: int = 32768    # Adjusted for Flash model's context window
    max_context_chunks: int = 60
    batch_size: int = 30       # Adjusted for Flash model
    parallel_questions: int = 15

    # Retrieval
    vector_top_k: int = 200
    similarity_threshold: float = 0.01  # Adjusted for Flash model

    # Generation
    temperature: float = 0.2    # Increased slightly for Flash model
    top_p: float = 0.9
    top_k: int = 32
    
    # Multilingual support
    default_language: str = "en"  # Default language code
    enable_language_detection: bool = True  # Auto-detect document language
    supported_languages: list = ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi"]  # Supported language codes
    
    # Security
    bearer_token: str = os.getenv("BEARER_TOKEN", "")
    
    # Production settings
    debug: bool = False
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()