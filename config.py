import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Keys - Switch to faster model
    openai_api_key: str = "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"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-3.5-turbo"  # Much faster than GPT-4
    pinecone_api_key: str = "pcsk_4kuoja_7z4urnnsUd8KZaiuxAsutoPrh9WmY8YXnDzUBzciTmJDjYWLDQoxQac4wHnhFFX"
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./hackrx_db.db")
    
    # Pinecone settings
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "hackrx-qa-index")
    
    # Processing settings - OPTIMIZED for speed and accuracy
    max_chunk_size: int = 800    # Smaller chunks for faster processing
    chunk_overlap: int = 100     # Reduced overlap for speed
    max_tokens: int = 150        # Much smaller for one-line answers
    
    # Security
    bearer_token: str = "433c9562217435ac71d779508405bfa9b20d0f58ff2aeb482c16c0e251f9f85f"
    
    class Config:
        env_file = ".env"

settings = Settings()