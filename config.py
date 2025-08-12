import os
from pydantic_settings import BaseSettings
from typing import Optional, List


class Settings(BaseSettings):
    # API Keys - Gemini 2.5 Flash optimized
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    # Embedding model configuration
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-base-en")
    embedding_cache_folder: str = os.getenv("EMBEDDING_CACHE_FOLDER", "./model_cache")
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./hackrx_db.db")
    
    # OPTIMIZED SETTINGS FOR GEMINI 2.5 FLASH
    # Context window: 1M input tokens, 65K output tokens
    
    # Chunking settings - optimized for 1M context window
    max_chunk_size: int = 2048              # Increased for better context utilization
    chunk_overlap: int = 256                # Proportionally increased overlap
    max_tokens: int = 1048576               # Full 1M token input capacity
    max_context_chunks: int = 400           # Significantly increased for 1M context
    parallel_questions: int = 30       # Increase from 15 to 30
    batch_size: int = 50               # Increase from 30 to 50
    parallel_questions: int = 25            # Increased for Gemini 2.5 Flash speed
    
    # Retrieval settings - optimized for large context
    vector_top_k: int = 100                 # Increased for better retrieval
    similarity_threshold: float = 0.003     # Slightly more selective
    rerank_top_k: int = 100                 # Increased for better ranking
    
    # Generation settings - optimized for Gemini 2.5 Flash
    temperature: float = 0.15               # Lower for more consistent RAG responses
    top_p: float = 0.95                     # Higher for better diversity
    top_k: int = 40                         # Slightly increased
    max_output_tokens: int = 32768          # Utilize ~50% of available 65K output tokens
    
    # Advanced RAG settings
    enable_hybrid_search: bool = True       # Combine dense and sparse retrieval
    enable_query_expansion: bool = True     # Expand queries for better retrieval
    enable_context_compression: bool = True # Important with large context
    
    # Document processing - optimized for large context
    enable_document_summarization: bool = True
    summary_chunk_size: int = 8192          # Larger chunks for better summaries
    enable_metadata_extraction: bool = True
    
    # Multilingual support
    default_language: str = "en"
    enable_language_detection: bool = True
    supported_languages: List[str] = [
        "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", 
        "ar", "hi", "bn", "ur", "tr", "pl", "nl", "sv", "da", "no"
    ]
    
    # Quality and reliability settings
    enable_response_validation: bool = True
    enable_hallucination_detection: bool = True
    confidence_threshold: float = 0.75      # Slightly higher for better quality
    max_retries: int = 3
    
    # Caching settings for performance
    enable_vector_cache: bool = True
    enable_response_cache: bool = True
    cache_ttl: int = 7200                   # Longer cache for stable responses
    
    # Rate limiting - optimized for Gemini 2.5 Flash speed
    requests_per_minute: int = 120          # Increased for faster model
    concurrent_requests: int = 15           # Increased for better throughput
    
    # Security
    bearer_token: str = os.getenv("BEARER_TOKEN", "")
    enable_input_sanitization: bool = True
    max_input_length: int = 900000          # Near 1M token limit
    
    # Monitoring and observability
    enable_metrics: bool = True
    enable_tracing: bool = False
    log_requests: bool = False
    
    # Production settings
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    environment: str = os.getenv("ENVIRONMENT", "production")
    
    # Performance optimization - leverage Gemini 2.5 Flash speed
    enable_streaming: bool = True
    stream_chunk_size: int = 2048           # Larger chunks for streaming
    
    # Fallback and error handling
    enable_fallback_model: bool = True
    fallback_model: str = "gemini-1.5-flash"
    max_processing_time: int = 180          # Reduced due to Flash speed
    
    # Cost optimization - updated for Gemini 2.5 Flash pricing
    enable_cost_tracking: bool = True
    max_daily_cost: float = 150.0          # Adjusted for Flash pricing
    input_token_cost: float = 0.15         # Per 1M input tokens
    output_token_cost: float = 0.60        # Per 1M output tokens
    thinking_token_cost: float = 3.50      # Per 1M thinking tokens
    
    # Gemini 2.5 Flash specific features
    enable_thinking_mode: bool = True       # Utilize thinking capabilities
    thinking_budget: str = "medium"         # Options: low, medium, high
    enable_native_tools: bool = True        # Enable native tool use
    enable_multimodal: bool = True          # Enable image/video/audio processing
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
        case_sensitive = False


# Create settings instance
settings = Settings()


def validate_gemini_flash_settings():
    """Validate settings specifically for Gemini 2.5 Flash."""
    warnings = []
    errors = []
    
    # Critical validations
    if not settings.gemini_api_key:
        errors.append("GEMINI_API_KEY is required for Gemini 2.5 Flash")
    
    # Gemini 2.5 Flash specific validations
    if settings.max_tokens > 1048576:
        errors.append("max_tokens exceeds Gemini 2.5 Flash's 1M input limit")
    
    if settings.max_output_tokens > 65535:
        errors.append("max_output_tokens exceeds Gemini 2.5 Flash's ~65K output limit")
    
    # Performance optimizations
    if settings.max_chunk_size < 1024:
        warnings.append("Consider larger chunk_size (2048+) to better utilize 1M context window")
    
    if settings.max_context_chunks < 200:
        warnings.append("max_context_chunks could be higher to leverage full 1M context")
    
    if settings.temperature > 0.3:
        warnings.append("Lower temperature recommended for RAG consistency")
    
    # Cost warnings
    if settings.max_output_tokens > 32768:
        warnings.append("High output tokens may increase costs significantly ($0.60/1M)")
    
    if settings.enable_thinking_mode and settings.thinking_budget == "high":
        warnings.append("High thinking budget increases costs to $3.50/1M tokens")
    
    # Context utilization
    estimated_context_usage = (settings.max_context_chunks * settings.max_chunk_size)
    context_utilization = (estimated_context_usage / settings.max_tokens) * 100
    
    if context_utilization < 50:
        warnings.append(f"Low context utilization ({context_utilization:.1f}%) - consider increasing chunk settings")
    
    return errors, warnings


# Performance recommendations
def get_performance_recommendations():
    """Get performance recommendations for Gemini 2.5 Flash."""
    recommendations = []
    
    recommendations.append("✅ Utilize the full 1M token context window for maximum RAG performance")
    recommendations.append("✅ Enable thinking mode for complex reasoning tasks")
    recommendations.append("✅ Use larger chunk sizes (2048+) to reduce fragmentation")
    recommendations.append("✅ Increase parallel processing to leverage Flash's speed")
    recommendations.append("✅ Consider multimodal capabilities for document processing")
    recommendations.append("⚠️  Monitor costs closely - output tokens are 4x more expensive than input")
    recommendations.append("⚠️  Thinking mode tokens cost 23x more than input tokens")
    
    return recommendations


# Auto-validate on import
if __name__ == "__main__":
    errors, warnings = validate_gemini_flash_settings()
    
    if errors:
        print("❌ Configuration Errors:")
        for error in errors:
            print(f"  • {error}")
    
    if warnings:
        print("⚠️  Configuration Warnings:")
        for warning in warnings:
            print(f"  • {warning}")
    
    print(f"\n🔧 Current Configuration:")
    print(f"Model: {settings.gemini_model}")
    print(f"Max input tokens: {settings.max_tokens:,}")
    print(f"Max output tokens: {settings.max_output_tokens:,}")
    print(f"Max chunks: {settings.max_context_chunks}")
    print(f"Chunk size: {settings.max_chunk_size}")
    
    estimated_usage = (settings.max_context_chunks * settings.max_chunk_size)
    utilization = (estimated_usage / settings.max_tokens) * 100
    print(f"Estimated context utilization: {utilization:.1f}%")
    
    print(f"\n💡 Performance Recommendations:")
    for rec in get_performance_recommendations():
        print(f"  {rec}")
    
    if not errors:
        print(f"\n✅ Configuration optimized for Gemini 2.5 Flash!")