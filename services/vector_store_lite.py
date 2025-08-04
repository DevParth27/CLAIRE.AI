import hashlib
import logging
from typing import List, Dict, Tuple
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class LightweightVectorStore:
    def __init__(self):
        # Enhanced TF-IDF for better semantic matching
        self.vectorizer = TfidfVectorizer(
            max_features=5000,  # Increased for better coverage
            stop_words='english',
            ngram_range=(1, 3),  # Include trigrams
            min_df=1,
            max_df=0.95,
            sublinear_tf=True   # Better scaling
        )
        self.documents = {}
        self.document_vectors = {}
        self.chunk_metadata = {}
        self.term_importance = {}
        
    async def store_document(self, chunks: List[Dict[str, str]], document_url: str) -> str:
        document_id = hashlib.md5(document_url.encode()).hexdigest()
        
        # Store chunks with enhanced metadata
        self.documents[document_id] = chunks
        self.chunk_metadata[document_id] = {
            chunk['chunk_id']: {
                'section_id': chunk.get('section_id', 0),
                'word_count': chunk.get('word_count', 0),
                'chunk_type': chunk.get('chunk_type', 'unknown'),
                'text_preview': chunk['text'][:150] + '...' if len(chunk['text']) > 150 else chunk['text'],
                'has_numbers': bool(re.search(r'\d+', chunk['text'])),
                'has_percentages': bool(re.search(r'\d+%', chunk['text'])),
                'has_time_periods': bool(re.search(r'\d+\s*(day|month|year)', chunk['text'], re.IGNORECASE))
            } for chunk in chunks
        }
        
        # Preprocess and create enhanced vectors
        processed_texts = [self._preprocess_text(chunk["text"]) for chunk in chunks]
        if processed_texts:
            vectors = self.vectorizer.fit_transform(processed_texts)
            self.document_vectors[document_id] = vectors
            
            # Build term importance mapping
            self._build_term_importance(processed_texts)
            
        logger.info(f"Stored {len(chunks)} chunks with enhanced semantic vectors and metadata")
        return document_id
    
    def _preprocess_text(self, text: str) -> str:
        """Advanced preprocessing for insurance policy documents"""
        # Normalize insurance-specific terms
        text = re.sub(r'\b(waiting|grace)\s+period', 'WAITING_PERIOD', text, flags=re.IGNORECASE)
        text = re.sub(r'\bpre-?existing\s+disease', 'PREEXISTING_DISEASE', text, flags=re.IGNORECASE)
        text = re.sub(r'\bno\s+claim\s+discount', 'NO_CLAIM_DISCOUNT', text, flags=re.IGNORECASE)
        text = re.sub(r'\broom\s+rent', 'ROOM_RENT', text, flags=re.IGNORECASE)
        text = re.sub(r'\bicu\s+charges', 'ICU_CHARGES', text, flags=re.IGNORECASE)
        text = re.sub(r'\bmaternity\s+(expenses?|benefits?)', 'MATERNITY_COVERAGE', text, flags=re.IGNORECASE)
        text = re.sub(r'\bhealth\s+check-?up', 'HEALTH_CHECKUP', text, flags=re.IGNORECASE)
        text = re.sub(r'\borgan\s+donor', 'ORGAN_DONOR', text, flags=re.IGNORECASE)
        text = re.sub(r'\bcataract\s+surgery', 'CATARACT_SURGERY', text, flags=re.IGNORECASE)
        text = re.sub(r'\bayush\s+treatment', 'AYUSH_TREATMENT', text, flags=re.IGNORECASE)
        
        # Normalize numerical patterns
        text = re.sub(r'\b\d+\s*%', 'PERCENTAGE_VALUE', text)
        text = re.sub(r'\b\d+\s*(days?)', 'DAYS_PERIOD', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d+\s*(months?)', 'MONTHS_PERIOD', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d+\s*(years?)', 'YEARS_PERIOD', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d+\s*(rupees?|rs\.?|inr)', 'AMOUNT_VALUE', text, flags=re.IGNORECASE)
        
        # Preserve important policy structure
        text = re.sub(r'\b(section|clause|article)\s+(\d+)', r'\1_\2', text, flags=re.IGNORECASE)
        
        return text
    
    def _build_term_importance(self, texts: List[str]):
        """Build term importance scores for better ranking"""
        # Get feature names and their IDF scores
        feature_names = self.vectorizer.get_feature_names_out()
        idf_scores = self.vectorizer.idf_
        
        self.term_importance = dict(zip(feature_names, idf_scores))
        
        # Boost importance of policy-specific terms
        policy_terms = [
            'WAITING_PERIOD', 'PREEXISTING_DISEASE', 'NO_CLAIM_DISCOUNT',
            'ROOM_RENT', 'ICU_CHARGES', 'MATERNITY_COVERAGE', 'HEALTH_CHECKUP',
            'ORGAN_DONOR', 'CATARACT_SURGERY', 'AYUSH_TREATMENT',
            'PERCENTAGE_VALUE', 'DAYS_PERIOD', 'MONTHS_PERIOD', 'YEARS_PERIOD'
        ]
        
        for term in policy_terms:
            if term in self.term_importance:
                self.term_importance[term] *= 1.5  # Boost policy-specific terms
    
    async def search_similar(self, query: str, document_id: str, top_k: int = 12) -> List[str]:
        """Enhanced similarity search with more results"""
        if document_id not in self.documents:
            return []
            
        try:
            # Preprocess the query for better matching
            processed_query = self._preprocess_text(query)
            query_vector = self.vectorizer.transform([processed_query])
            
            # Get document vectors
            if document_id not in self.document_vectors:
                logger.warning(f"Document vectors for {document_id} not found")
                return [chunk["text"] for chunk in self.documents[document_id][:top_k]]
            
            # Multi-strategy ranking
            ranked_chunks = self._multi_strategy_ranking(query, processed_query, query_vector, document_id)
            
            # Get top results with lower threshold for more comprehensive retrieval
            results = []
            used_texts = set()
            
            for idx, score in ranked_chunks:
                if score >= settings.similarity_threshold:
                    chunk_text = self.documents[document_id][idx]["text"]
                    if chunk_text not in used_texts:
                        results.append(chunk_text)
                        used_texts.add(chunk_text)
                        if len(results) >= settings.vector_top_k:
                            break
            
            # Ensure minimum context
            if len(results) < 5:
                for idx, _ in ranked_chunks:
                    chunk_text = self.documents[document_id][idx]["text"]
                    if chunk_text not in used_texts:
                        results.append(chunk_text)
                        used_texts.add(chunk_text)
                        if len(results) >= 5:
                            break
            
            logger.info(f"Found {len(results)} relevant chunks for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error in search_similar: {str(e)}")
            # Fallback to basic retrieval
            return [chunk["text"] for chunk in self.documents[document_id][:min(top_k, len(self.documents[document_id]))]]  
    
    def _multi_strategy_ranking(self, original_query: str, processed_query: str, query_vector, document_id: str) -> List[Tuple[int, float]]:
        """Advanced multi-strategy ranking combining multiple signals"""
        chunks = self.documents[document_id]
        query_lower = original_query.lower()
        
        # Strategy 1: Enhanced TF-IDF similarity (weight: 0.5)
        similarities = cosine_similarity(query_vector, self.document_vectors[document_id])
        tfidf_scores = similarities[0] * 0.5
        
        # Strategy 2: Exact phrase matching (weight: 0.25)
        phrase_scores = np.zeros(len(chunks))
        query_phrases = self._extract_phrases(original_query)
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk["text"].lower()
            phrase_matches = sum(1 for phrase in query_phrases if phrase in chunk_text)
            phrase_scores[i] = (phrase_matches / max(len(query_phrases), 1)) * 0.25
        
        # Strategy 3: Keyword density scoring (weight: 0.15)
        keyword_scores = np.zeros(len(chunks))
        query_keywords = set(re.findall(r'\b\w+\b', query_lower))
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk["text"].lower()
            chunk_words = re.findall(r'\b\w+\b', chunk_text)
            
            if chunk_words:
                keyword_matches = sum(1 for word in chunk_words if word in query_keywords)
                keyword_density = keyword_matches / len(chunk_words)
                keyword_scores[i] = keyword_density * 0.15
        
        # Strategy 4: Metadata-based scoring (weight: 0.1)
        metadata_scores = np.zeros(len(chunks))
        
        for i, chunk in enumerate(chunks):
            metadata = self.chunk_metadata[document_id][chunk['chunk_id']]
            score = 0
            
            # Boost chunks with numbers if query contains numerical terms
            if any(term in query_lower for term in ['period', 'days', 'months', 'years', '%', 'percent']):
                if metadata['has_numbers'] or metadata['has_percentages'] or metadata['has_time_periods']:
                    score += 0.05
            
            # Boost complete sections over split sections
            if metadata['chunk_type'] == 'complete_section':
                score += 0.02
            
            # Boost chunks with policy-specific indicators
            chunk_text = chunk["text"].lower()
            policy_indicators = ['coverage', 'benefit', 'waiting', 'grace', 'claim', 'premium', 'policy']
            indicator_count = sum(1 for indicator in policy_indicators if indicator in chunk_text)
            score += (indicator_count / len(policy_indicators)) * 0.03
            
            metadata_scores[i] = score
        
        # Combine all scores
        final_scores = tfidf_scores + phrase_scores + keyword_scores + metadata_scores
        
        # Apply minimum threshold and sort
        ranked_indices = final_scores.argsort()[::-1]
        return [(idx, final_scores[idx]) for idx in ranked_indices if final_scores[idx] > 0.05]
    
    def _extract_phrases(self, query: str) -> List[str]:
        """Extract meaningful phrases from query"""
        phrases = []
        
        # Extract quoted phrases
        quoted_phrases = re.findall(r'"([^"]+)"', query)
        phrases.extend(quoted_phrases)
        
        # Extract common insurance phrases
        insurance_phrases = [
            'waiting period', 'grace period', 'pre-existing disease', 'no claim discount',
            'room rent', 'icu charges', 'maternity expenses', 'health check-up',
            'organ donor', 'cataract surgery', 'ayush treatment'
        ]
        
        query_lower = query.lower()
        for phrase in insurance_phrases:
            if phrase in query_lower:
                phrases.append(phrase)
        
        # Extract numerical phrases
        numerical_phrases = re.findall(r'\d+\s*(?:days?|months?|years?|%|percent)', query_lower)
        phrases.extend(numerical_phrases)
        
        return list(set(phrases))  # Remove duplicates
    
    def _get_section_context(self, document_id: str, chunk_idx: int, section_id: int) -> str:
        """Get expanded context from the same section"""
        chunks = self.documents[document_id]
        section_chunks = []
        
        # Find all chunks from the same section
        for i, chunk in enumerate(chunks):
            if self.chunk_metadata[document_id][chunk['chunk_id']]['section_id'] == section_id:
                section_chunks.append((i, chunk['text']))
        
        # If we have multiple chunks from same section, combine them intelligently
        if len(section_chunks) > 1:
            # Sort by chunk index to maintain order
            section_chunks.sort(key=lambda x: x[0])
            
            # Find the target chunk and include neighbors
            target_pos = next(i for i, (idx, _) in enumerate(section_chunks) if idx == chunk_idx)
            
            context_parts = []
            start_pos = max(0, target_pos - 1)
            end_pos = min(len(section_chunks), target_pos + 2)
            
            for i in range(start_pos, end_pos):
                _, text = section_chunks[i]
                if i == target_pos:
                    context_parts.append(f"[MAIN] {text}")
                else:
                    context_parts.append(f"[CONTEXT] {text[:200]}..." if len(text) > 200 else f"[CONTEXT] {text}")
            
            return " ".join(context_parts)
        else:
            return chunks[chunk_idx]['text']
    
    def _keyword_fallback_search(self, query: str, document_id: str, top_k: int) -> List[str]:
        """Enhanced fallback search using multiple keyword strategies"""
        query_words = set(query.lower().split())
        chunks = self.documents[document_id]
        
        scored_chunks = []
        
        for chunk in chunks:
            chunk_text = chunk["text"].lower()
            chunk_words = set(chunk_text.split())
            
            # Calculate multiple similarity metrics
            word_overlap = len(query_words.intersection(chunk_words))
            jaccard_similarity = word_overlap / len(query_words.union(chunk_words)) if query_words.union(chunk_words) else 0
            
            # Boost score for exact phrase matches
            phrase_bonus = 0
            for phrase in self._extract_phrases(query):
                if phrase.lower() in chunk_text:
                    phrase_bonus += 0.5
            
            total_score = word_overlap + jaccard_similarity + phrase_bonus
            
            if total_score > 0:
                scored_chunks.append((chunk["text"], total_score))
        
        # Sort by score and return top results
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk[0] for chunk in scored_chunks[:top_k]]