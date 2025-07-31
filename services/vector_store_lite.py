import hashlib
import logging
from typing import List, Dict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)

class LightweightVectorStore:
    def __init__(self):
        # Use TF-IDF instead of sentence transformers (much lighter)
        self.vectorizer = TfidfVectorizer(
            max_features=1000,  # Limit features for memory
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.documents = {}
        self.document_vectors = {}
        
    async def store_document(self, chunks: List[Dict[str, str]], document_url: str) -> str:
        document_id = hashlib.md5(document_url.encode()).hexdigest()
        
        # Store chunks
        self.documents[document_id] = chunks
        
        # Create TF-IDF vectors
        texts = [chunk["text"] for chunk in chunks]
        if texts:
            vectors = self.vectorizer.fit_transform(texts)
            self.document_vectors[document_id] = vectors
            
        logger.info(f"Stored {len(chunks)} chunks with lightweight vectors")
        return document_id
    
    async def search_similar(self, query: str, document_id: str, top_k: int = 3) -> List[str]:  # Reduced from 5
        """Fast similarity search with reduced results"""
        if document_id not in self.documents:
            return []
            
        # Transform query
        try:
            query_vector = self.vectorizer.transform([query])
        except:
            # If vectorizer not fitted, return first few chunks
            return [chunk["text"] for chunk in self.documents[document_id][:top_k]]
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.document_vectors[document_id])
        
        # Get top results (reduced number)
        top_indices = similarities[0].argsort()[-top_k:][::-1]
        
        return [self.documents[document_id][i]["text"][:500] for i in top_indices]  # Truncate chunks