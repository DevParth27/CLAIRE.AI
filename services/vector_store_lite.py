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
        # Enhanced TF-IDF for better semantic matching
        self.vectorizer = TfidfVectorizer(
            max_features=3000,  # Increased for better coverage
            stop_words='english',
            ngram_range=(1, 3),  # Include trigrams
            min_df=1,
            max_df=0.95
        )
        self.documents = {}
        self.document_vectors = {}
        
    async def store_document(self, chunks: List[Dict[str, str]], document_url: str) -> str:
        document_id = hashlib.md5(document_url.encode()).hexdigest()
        
        # Store chunks
        self.documents[document_id] = chunks
        
        # Create enhanced TF-IDF vectors
        texts = [chunk["text"] for chunk in chunks]
        if texts:
            vectors = self.vectorizer.fit_transform(texts)
            self.document_vectors[document_id] = vectors
            
        logger.info(f"Stored {len(chunks)} chunks with enhanced vectors")
        return document_id
    
    async def search_similar(self, query: str, document_id: str, top_k: int = 6) -> List[str]:
        """Enhanced similarity search with more results"""
        if document_id not in self.documents:
            return []
            
        # Transform query
        try:
            query_vector = self.vectorizer.transform([query])
        except:
            # Return more chunks if vectorizer not fitted
            return [chunk["text"] for chunk in self.documents[document_id][:top_k]]
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.document_vectors[document_id])
        
        # Get top results with higher threshold
        similarity_scores = similarities[0]
        top_indices = similarity_scores.argsort()[-top_k:][::-1]
        
        # Filter by minimum similarity threshold
        filtered_results = []
        for idx in top_indices:
            if similarity_scores[idx] > 0.1:  # Minimum relevance threshold
                # Don't truncate chunks - preserve full context
                filtered_results.append(self.documents[document_id][idx]["text"])
        
        # If no good matches, return top chunks anyway
        if not filtered_results:
            filtered_results = [self.documents[document_id][i]["text"] for i in top_indices[:3]]
            
        return filtered_results