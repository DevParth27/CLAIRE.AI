import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Keys - Updated for OpenRouter
    openai_api_key: str = "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"
    openai_base_url: str = "https://openrouter.ai/api/v1"
    openai_model: str = "deepseek/deepseek-r1-0528-qwen3-8b:free"
    pinecone_api_key: str = "pcsk_4kuoja_7z4urnnsUd8KZaiuxAsutoPrh9WmY8YXnDzUBzciTmJDjYWLDQoxQac4wHnhFFX"
    
    # Database - Changed to SQLite
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./hackrx_db.db")
    
    # Pinecone settings
    pinecone_environment: str = "us-east-1-aws"
    pinecone_index_name: str = "hackrx-qa-index"
    
    # Processing settings
    max_chunk_size: int = 1000
    chunk_overlap: int = 200
    max_tokens: int = 4000
    
    # Security
    bearer_token: str = "433c9562217435ac71d779508405bfa9b20d0f58ff2aeb482c16c0e251f9f85f"
    
    class Config:
        env_file = ".env"

settings = Settings()