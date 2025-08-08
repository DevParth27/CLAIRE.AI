# import os
# from pydantic_settings import BaseSettings
# from typing import Optional

# class Settings(BaseSettings):
#     # API Keys - Using Gemini 2.5 Flash
#     gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
#     gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    
#     # Database
#     database_url: str = os.getenv("DATABASE_URL", "sqlite:///./hackrx_db.db")
    
#     # Enhanced processing settings for Gemini-2.5-flash
#     # Dynamic Document Parsing RAG Configuration - Maximum Accuracy

#     # Chunking settings - optimized for Gemini-2.5-flash
#     max_chunk_size: int = 768   # Adjusted for Flash model's context window
#     chunk_overlap: int = 192    # Adjusted for Flash model
#     max_tokens: int = 524288    # Maximum input tokens for Gemini-2.5-flash
#     max_context_chunks: int = 80  # Adjusted for Flash model
#     batch_size: int = 30        # Adjusted for Flash model
#     parallel_questions: int = 15  # Adjusted for Flash model

#     # Retrieval
#     vector_top_k: int = 200      # Adjusted for Flash model
#     similarity_threshold: float = 0.005  # Kept the same

#     # Generation
#     temperature: float = 0.2     # Slightly increased for Flash model
#     top_p: float = 0.9           # Adjusted for Flash model
#     top_k: int = 32              # Adjusted for Flash model
    
#     # Multilingual support
#     default_language: str = "en"  # Default language code
#     enable_language_detection: bool = True  # Auto-detect document language
#     supported_languages: list = ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi"]  # Supported language codes
    
#     # Security
#     bearer_token: str = os.getenv("BEARER_TOKEN", "")
    
#     # Production settings
#     debug: bool = False
#     log_level: str = "INFO"
    
#     class Config:
#         env_file = ".env"
#         extra = "ignore"

# settings = Settings()

import os
from pydantic_settings import BaseSettings
from typing import Optional, List


class Settings(BaseSettings):
    # API Keys - Using Gemini 2.5 Flash
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    # Embedding model configuration
    # Options: "intfloat/e5-large-v2", "jinaai/jina-embeddings-v2-base-en", "all-MiniLM-L6-v2"
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "intfloat/e5-large-v2")
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./hackrx_db.db")
    
    # Enhanced processing settings optimized for Gemini 2.5 Flash
    # Dynamic Document Parsing RAG Configuration
    
    # Chunking settings - optimized for Gemini 2.5 Flash's context window
    max_chunk_size: int = 768          # Optimized for Flash model
    chunk_overlap: int = 192           # Optimized for Flash model
    max_tokens: int = 524288           # Maximum input tokens for Gemini 2.5 Flash
    max_context_chunks: int = 80       # Adjusted for Flash model
    batch_size: int = 30               # Adjusted for Flash model
    parallel_questions: int = 15       # Adjusted for Flash model
    
    # Retrieval settings - optimized for Flash model
    vector_top_k: int = 200            # Adjusted for Flash model
    similarity_threshold: float = 0.005 # Adjusted for Flash model
    rerank_top_k: int = 50             # Adjusted for Flash model
    
    # Generation settings - optimized for Gemini 2.5 Flash
    temperature: float = 0.2           # Slightly higher for Flash model
    top_p: float = 0.9                 # Adjusted for Flash model
    top_k: int = 32                    # Adjusted for Flash model
    max_output_tokens: int = 4096      # Flash model's output limit
    
    # Advanced RAG settings
    enable_hybrid_search: bool = True    # Combine dense and sparse retrieval
    enable_query_expansion: bool = True  # Expand queries for better retrieval
    enable_context_compression: bool = True  # Compress context for efficiency
    
    # Document processing
    enable_document_summarization: bool = True
    summary_chunk_size: int = 4000      # For document summarization
    enable_metadata_extraction: bool = True
    
    # Multilingual support - expanded for better coverage
    default_language: str = "en"
    enable_language_detection: bool = True
    supported_languages: List[str] = [
        "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", 
        "ar", "hi", "bn", "ur", "tr", "pl", "nl", "sv", "da", "no"
    ]
    
    # Quality and reliability settings
    enable_response_validation: bool = True
    enable_hallucination_detection: bool = True
    confidence_threshold: float = 0.7
    max_retries: int = 3
    
    # Caching settings for performance
    enable_vector_cache: bool = True
    enable_response_cache: bool = True
    cache_ttl: int = 3600              # Cache TTL in seconds
    
    # Rate limiting and throttling
    requests_per_minute: int = 60      # Adjust based on your quota
    concurrent_requests: int = 10      # Number of concurrent requests
    
    # Security
    bearer_token: str = os.getenv("BEARER_TOKEN", "")
    enable_input_sanitization: bool = True
    max_input_length: int = 100000     # Maximum input length
    
    # Monitoring and observability
    enable_metrics: bool = True
    enable_tracing: bool = False       # Enable for debugging
    log_requests: bool = False         # Enable for debugging
    
    # Production settings
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    environment: str = os.getenv("ENVIRONMENT", "production")
    
    # Performance optimization
    enable_streaming: bool = True      # Enable streaming responses
    stream_chunk_size: int = 1024     # Streaming chunk size
    
    # Fallback and error handling
    enable_fallback_model: bool = True
    fallback_model: str = "gemini-1.5-flash"  # Faster fallback
    max_processing_time: int = 300     # Maximum processing time in seconds
    
    # Cost optimization
    enable_cost_tracking: bool = True
    max_daily_cost: float = 100.0     # Maximum daily cost limit
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
        case_sensitive = False


# Create settings instance
settings = Settings()


# Validation function to ensure settings are properly configured
def validate_settings():
    """Validate critical settings and provide warnings for suboptimal configurations."""
    warnings = []
    errors = []
    
    # Critical validations
    if not settings.gemini_api_key:
        errors.append("GEMINI_API_KEY is required")
    
    # Performance warnings
    if settings.max_chunk_size < 512:
        warnings.append("max_chunk_size is quite small, consider increasing for better context")
    
    if settings.vector_top_k > settings.max_context_chunks * 2:
        warnings.append("vector_top_k is much larger than max_context_chunks, this may be inefficient")
    
    if settings.temperature > 0.5:
        warnings.append("High temperature may reduce consistency in RAG responses")
    
    # Resource warnings
    if settings.max_tokens > 2000000:
        warnings.append("max_tokens exceeds Gemini 1.5 Pro's actual limit")
    
    if settings.concurrent_requests > 20:
        warnings.append("High concurrent requests may hit rate limits")
    
    return errors, warnings


# Auto-validate on import
if __name__ == "__main__":
    errors, warnings = validate_settings()
    
    if errors:
        print("❌ Configuration Errors:")
        for error in errors:
            print(f"  • {error}")
    
    if warnings:
        print("⚠️  Configuration Warnings:")
        for warning in warnings:
            print(f"  • {warning}")
    
    if not errors and not warnings:
        print("✅ Configuration looks good!")
    
    print(f"\nCurrent model: {settings.gemini_model}")
    print(f"Max context tokens: {settings.max_tokens:,}")
    print(f"Max chunks: {settings.max_context_chunks}")
    print(f"Environment: {settings.environment}")