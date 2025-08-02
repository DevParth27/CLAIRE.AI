import hashlib
import logging
from typing import List, Dict, Tuple
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)

class LightweightVectorStore:
    def __init__(self):
        # Enhanced TF-IDF with better parameters
        self.vectorizer = TfidfVectorizer(
            max_features=5000,  # Increased vocabulary
            stop_words='english',
            ngram_range=(1, 4),  # Include 4-grams for better phrase matching
            min_df=1,
            max_df=0.9,
            sublinear_tf=True,  # Better handling of term frequencies
            norm='l2'  # L2 normalization
        )
        self.documents = {}
        self.document_vectors = {}
        self.chunk_metadata = {}
        
    async def store_document(self, chunks: List[Dict[str, str]], document_url: str) -> str:
        document_id = hashlib.md5(document_url.encode()).hexdigest()
        
        # Store chunks with metadata
        self.documents[document_id] = chunks
        self.chunk_metadata[document_id] = {
            chunk['chunk_id']: {
                'section_id': chunk.get('section_id', 0),
                'word_count': chunk.get('word_count', 0),
                'text_preview': chunk['text'][:100] + '...' if len(chunk['text']) > 100 else chunk['text']
            } for chunk in chunks
        }
        
        # Create enhanced vectors with preprocessing
        texts = [self._preprocess_text(chunk["text"]) for chunk in chunks]
        if texts:
            vectors = self.vectorizer.fit_transform(texts)
            self.document_vectors[document_id] = vectors
            
        logger.info(f"Stored {len(chunks)} chunks with enhanced semantic vectors")
        return document_id
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better vectorization"""
        # Normalize insurance/policy specific terms
        text = re.sub(r'\b(premium|policy|coverage|benefit|claim)s?\b', 
                     lambda m: m.group().lower(), text, flags=re.IGNORECASE)
        
        # Normalize numbers and percentages
        text = re.sub(r'\b\d+%\b', 'PERCENTAGE', text)
        text = re.sub(r'\b\d+\s*(days?|months?|years?)\b', 'TIMEPERIOD', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d+\s*(rupees?|rs\.?|inr)\b', 'AMOUNT', text, flags=re.IGNORECASE)
        
        return text
    
    async def search_similar(self, query: str, document_id: str, top_k: int = 8) -> List[str]:
        """Enhanced similarity search with multiple strategies"""
        if document_id not in self.documents:
            return []
        
        try:
            # Preprocess query
            processed_query = self._preprocess_text(query)
            query_vector = self.vectorizer.transform([processed_query])
        except:
            # Fallback: return chunks based on keyword matching
            return self._keyword_fallback_search(query, document_id, top_k)
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.document_vectors[document_id])
        similarity_scores = similarities[0]
        
        # Multi-strategy ranking
        ranked_chunks = self._multi_strategy_ranking(query, similarity_scores, document_id)
        
        # Return top results with context
        results = []
        for chunk_idx, score in ranked_chunks[:top_k]:
            chunk_text = self.documents[document_id][chunk_idx]["text"]
            
            # Add context from adjacent chunks if score is high
            if score > 0.3 and len(results) < 5:
                context_text = self._get_contextual_chunk(document_id, chunk_idx)
                results.append(context_text)
            else:
                results.append(chunk_text)
        
        # Ensure we have some results
        if not results:
            results = [chunk["text"] for chunk in self.documents[document_id][:3]]
            
        return results
    
    def _multi_strategy_ranking(self, query: str, similarity_scores: np.ndarray, document_id: str) -> List[Tuple[int, float]]:
        """Combine multiple ranking strategies"""
        query_lower = query.lower()
        chunks = self.documents[document_id]
        
        # Strategy 1: TF-IDF similarity (weight: 0.6)
        tfidf_scores = similarity_scores * 0.6
        
        # Strategy 2: Keyword matching (weight: 0.3)
        keyword_scores = np.zeros(len(chunks))
        query_keywords = set(re.findall(r'\b\w+\b', query_lower))
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk["text"].lower()
            chunk_keywords = set(re.findall(r'\b\w+\b', chunk_text))
            keyword_overlap = len(query_keywords.intersection(chunk_keywords))
            keyword_scores[i] = keyword_overlap / max(len(query_keywords), 1) * 0.3
        
        # Strategy 3: Section relevance (weight: 0.1)
        section_scores = np.zeros(len(chunks))
        for i, chunk in enumerate(chunks):
            # Boost chunks from sections that might contain policy details
            chunk_text = chunk["text"].lower()
            if any(term in chunk_text for term in ['waiting period', 'coverage', 'benefit', 'premium', 'claim']):
                section_scores[i] = 0.1
        
        # Combine scores
        final_scores = tfidf_scores + keyword_scores + section_scores
        
        # Create ranked list
        ranked_indices = final_scores.argsort()[::-1]
        return [(idx, final_scores[idx]) for idx in ranked_indices if final_scores[idx] > 0.05]
    
    def _get_contextual_chunk(self, document_id: str, chunk_idx: int) -> str:
        """Get chunk with surrounding context"""
        chunks = self.documents[document_id]
        
        # Include previous and next chunk for context
        context_chunks = []
        
        if chunk_idx > 0:
            prev_chunk = chunks[chunk_idx - 1]["text"]
            if len(prev_chunk) < 200:  # Only add if previous chunk is short
                context_chunks.append(prev_chunk[-100:])  # Last 100 chars
        
        context_chunks.append(chunks[chunk_idx]["text"])
        
        if chunk_idx < len(chunks) - 1:
            next_chunk = chunks[chunk_idx + 1]["text"]
            if len(next_chunk) < 200:  # Only add if next chunk is short
                context_chunks.append(next_chunk[:100])  # First 100 chars
        
        return " ... ".join(context_chunks)
    
    def _keyword_fallback_search(self, query: str, document_id: str, top_k: int) -> List[str]:
        """Fallback search using keyword matching"""
        query_words = set(query.lower().split())
        chunks = self.documents[document_id]
        
        scored_chunks = []
        for chunk in chunks:
            chunk_words = set(chunk["text"].lower().split())
            overlap = len(query_words.intersection(chunk_words))
            if overlap > 0:
                scored_chunks.append((chunk["text"], overlap))
        
        # Sort by overlap score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        return [chunk[0] for chunk in scored_chunks[:top_k]]