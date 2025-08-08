import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from config import settings

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Utility class to manage different embedding models"""
    
    def __init__(self):
        self.model_name = settings.embedding_model
        self.model = self._initialize_model()
        self.dimension = self._get_dimension()
        logger.info(f"EmbeddingManager initialized with model: {self.model_name}")
    
    def _initialize_model(self) -> Any:
        """Initialize the appropriate embedding model"""
        try:
            # For Sentence Transformers models
            return SentenceTransformer(self.model_name)
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            logger.warning("Falling back to default embedding model")
            return SentenceTransformer('all-MiniLM-L6-v2')
    
    def _get_dimension(self) -> int:
        """Get the dimension of the embedding model"""
        if "e5-large-v2" in self.model_name:
            return 1024
        elif "jina-embeddings-v2-base-en" in self.model_name:
            return 768
        else:
            # Default for all-MiniLM-L6-v2
            return 384
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts"""
        try:
            # Sentence Transformers
            return self.model.encode(texts, convert_to_numpy=True).tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise