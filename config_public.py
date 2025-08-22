import os
from pydantic_settings import BaseSettings
from typing import Optional, List

class Settings(BaseSettings):
    """Public configuration - Sensitive details removed"""
    
    # API Keys - Placeholder only
    gemini_api_key: str = "[REQUIRED - CONTACT_DEVELOPER]"
    gemini_model: str = "[OPTIMIZED_MODEL_DETAILS_AVAILABLE_ON_REQUEST]"
    
    # Basic settings only
    embedding_model: str = "[CUSTOM_EMBEDDING_SOLUTION]"
    embedding_cache_folder: str = "./model_cache"
    
    # Database
    database_url: str = "sqlite:///./demo.db"
    
    # Reduced capabilities for public version
    max_chunk_size: int = 1024  # Reduced from optimized value
    chunk_overlap: int = 128    # Reduced from optimized value
    max_tokens: int = 50000     # Significantly reduced
    max_context_chunks: int = 20  # Reduced from optimized value
    
    # Basic retrieval settings
    vector_top_k: int = 10      # Reduced from optimized value
    similarity_threshold: float = 0.1  # Less selective
    
    # Generation settings - Basic only
    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 20
    max_output_tokens: int = 1000  # Significantly reduced
    
    # Advanced features disabled in public version
    enable_hybrid_search: bool = False
    enable_query_expansion: bool = False
    enable_context_compression: bool = False
    enable_document_summarization: bool = False
    
    # Security - Placeholder
    bearer_token: str = "[CONTACT_FOR_PRODUCTION_TOKEN]"
    
    class Config:
        env_file = ".env.example"
        env_file_encoding = "utf-8"
        extra = "ignore"
        case_sensitive = False

# Note: This is a limited public version
# Full optimized configuration available under commercial license
settings = Settings()

def get_demo_limitations():
    return [
        "⚠️  Limited to basic functionality",
        "⚠️  Advanced AI optimizations not included",
        "⚠️  Production-grade features require license",
        "⚠️  Contact developer for full implementation"
    ]