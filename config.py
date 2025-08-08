import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Keys - Using Gemini 2.5 Flash
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./hackrx_db.db")
    
    # Enhanced processing settings for Gemini-2.5-flash
    # Dynamic Document Parsing RAG Configuration - Maximum Accuracy

    # Chunking settings - optimized for Gemini-2.5-flash
    max_chunk_size: int = 768   # Adjusted for Flash model's context window
    chunk_overlap: int = 192    # Adjusted for Flash model
    max_tokens: int = 524288    # Maximum input tokens for Gemini-2.5-flash
    max_context_chunks: int = 80  # Adjusted for Flash model
    batch_size: int = 30        # Adjusted for Flash model
    parallel_questions: int = 15  # Adjusted for Flash model

    # Retrieval
    vector_top_k: int = 200      # Adjusted for Flash model
    similarity_threshold: float = 0.005  # Kept the same

    # Generation
    temperature: float = 0.2     # Slightly increased for Flash model
    top_p: float = 0.9           # Adjusted for Flash model
    top_k: int = 32              # Adjusted for Flash model
    
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