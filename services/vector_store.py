import hashlib
import logging
import re
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import settings
import os

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        # Initialize embedding model based on configuration
        self.embedding_model_name = settings.embedding_model
        self.embedding_model = self._initialize_embedding_model()
        
        # Set embedding dimension based on model
        if "e5-large-v2" in self.embedding_model_name:
            self.embedding_dimension = 1024
        elif "jina-embeddings-v2-base-en" in self.embedding_model_name:
            self.embedding_dimension = 768
        else:
            # Default for all-MiniLM-L6-v2
            self.embedding_dimension = 384
        
        # In-memory storage for development
        self.vectors = {}  # {vector_id: {"embedding": [...], "metadata": {...}}}
        self.documents = {}  # {document_id: [vector_ids]}
        self.chunk_metadata = {}  # Enhanced metadata storage
        
        logger.info(f"Initialized optimized in-memory vector store with {self.embedding_model_name} embeddings")

    def _initialize_embedding_model(self):
        """Initialize the appropriate embedding model based on configuration"""
        try:
            # For Sentence Transformers models with cache folder
            return SentenceTransformer(self.embedding_model_name, cache_folder=settings.embedding_cache_folder)
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            logger.warning("Falling back to default embedding model")
            return SentenceTransformer('all-MiniLM-L6-v2', cache_folder=settings.embedding_cache_folder)
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts using the configured model"""
        try:
            # Process in batches to avoid memory issues
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_embeddings = self.embedding_model.encode(batch, convert_to_numpy=True).tolist()
                all_embeddings.extend(batch_embeddings)
                
            return all_embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def _ensure_index_exists(self):
        """Mock method - no index creation needed for in-memory storage"""
        logger.info("Using in-memory vector storage")
    
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
    
    async def store_document(self, chunks: List[Dict[str, str]], document_url: str) -> str:
        """Store document chunks in in-memory vector database with enhanced metadata"""
        try:
            # Generate document ID
            document_id = hashlib.md5(document_url.encode()).hexdigest()
            
            # Store vectors
            vector_ids = []
            
            # Store chunks with enhanced metadata
            self.chunk_metadata[document_id] = {}
            
            for chunk in chunks:
                # Preprocess text for better embedding
                processed_text = self._preprocess_text(chunk["text"])
                
                # Generate embedding
                embedding = self.embedding_model.encode(processed_text)
                
                # Create vector ID
                vector_id = f"{document_id}_{chunk['chunk_id']}"
                
                # Prepare metadata
                metadata = {
                    "document_id": document_id,
                    "document_url": document_url,
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "start_word": chunk.get("start_word", 0),
                    "end_word": chunk.get("end_word", 0),
                    "section_id": chunk.get("section_id", 0),
                    "word_count": len(chunk["text"].split()),
                    "chunk_type": chunk.get("chunk_type", "unknown"),
                    "has_numbers": bool(re.search(r'\d+', chunk["text"])),
                    "has_percentages": bool(re.search(r'\d+%', chunk["text"])),
                    "has_time_periods": bool(re.search(r'\d+\s*(day|month|year)', chunk["text"], re.IGNORECASE))
                }
                
                # Store in memory
                self.vectors[vector_id] = {
                    "embedding": embedding,
                    "metadata": metadata
                }
                vector_ids.append(vector_id)
                
                # Store enhanced metadata for faster access
                self.chunk_metadata[document_id][chunk['chunk_id']] = {
                    "section_id": metadata["section_id"],
                    "word_count": metadata["word_count"],
                    "chunk_type": metadata["chunk_type"],
                    "text_preview": chunk["text"][:150] + '...' if len(chunk["text"]) > 150 else chunk["text"],
                    "has_numbers": metadata["has_numbers"],
                    "has_percentages": metadata["has_percentages"],
                    "has_time_periods": metadata["has_time_periods"]
                }
            
            # Track document vectors
            self.documents[document_id] = vector_ids
            
            logger.info(f"Stored {len(chunks)} chunks for document {document_id} in memory with enhanced metadata")
            return document_id
            
        except Exception as e:
            logger.error(f"Error storing document: {str(e)}")
            raise

    def _extract_phrases(self, query: str) -> List[str]:
        """Extract meaningful phrases from query"""
        phrases = []
        
        # Extract quoted phrases
        quoted_phrases = re.findall(r'"([^"]+)"', query)
        phrases.extend(quoted_phrases)
        
        # Extract common insurance phrases
        insurance_phrases = [
            "waiting period", "grace period", "pre-existing disease", "no claim discount",
            "room rent", "icu charges", "maternity expenses", "health check-up",
            "organ donor", "cataract surgery", "ayush treatment"
        ]
        
        query_lower = query.lower()
        for phrase in insurance_phrases:
            if phrase in query_lower:
                phrases.append(phrase)
        
        # Extract numerical phrases
        numerical_phrases = re.findall(r'\d+\s*(?:days?|months?|years?|%|percent)', query_lower)
        phrases.extend(numerical_phrases)
        
        return list(set(phrases))  # Remove duplicates

    async def search_similar(self, query: str, document_id: str, top_k: int = None) -> List[str]:
        """Comprehensive search with multiple strategies to find ALL relevant information"""
        try:
            if top_k is None:
                top_k = settings.vector_top_k  # Use the value from config
                
            # Get document vectors
            if document_id not in self.documents:
                logger.warning(f"Document {document_id} not found")
                return []
            
            # Preprocess the query for better matching
            processed_query = self._preprocess_text(query)
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(processed_query)
            
            # Calculate similarities
            similarities = []
            for vector_id in self.documents[document_id]:
                if vector_id in self.vectors:
                    vector_data = self.vectors[vector_id]
                    similarity = cosine_similarity(
                        [query_embedding], 
                        [vector_data["embedding"]]
                    )[0][0]
                    
                    chunk_id = vector_data["metadata"]["chunk_id"]
                    
                    # Apply metadata-based scoring adjustments
                    metadata_score = 0
                    if document_id in self.chunk_metadata and chunk_id in self.chunk_metadata[document_id]:
                        metadata = self.chunk_metadata[document_id][chunk_id]
                        
                        # Boost chunks with numbers if query contains numerical terms
                        query_lower = query.lower()
                        if any(term in query_lower for term in ['period', 'days', 'months', 'years', '%', 'percent']):
                            if metadata['has_numbers'] or metadata['has_percentages'] or metadata['has_time_periods']:
                                metadata_score += 0.05
                        
                        # Boost complete sections over split sections
                        if metadata['chunk_type'] == 'complete_section':
                            metadata_score += 0.02
                    
                    # Apply phrase matching bonus
                    phrase_bonus = 0
                    query_phrases = self._extract_phrases(query)
                    text_lower = vector_data["metadata"]["text"].lower()
                    phrase_matches = sum(1 for phrase in query_phrases if phrase.lower() in text_lower)
                    if phrase_matches > 0:
                        phrase_bonus = (phrase_matches / max(len(query_phrases), 1)) * 0.1
                    
                    # Combine scores
                    adjusted_score = similarity + metadata_score + phrase_bonus
                    
                    similarities.append({
                        "score": adjusted_score,
                        "metadata": vector_data["metadata"],
                        "text": vector_data["metadata"]["text"]
                    })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x["score"], reverse=True)
            
            # Multi-strategy comprehensive retrieval
            relevant_chunks = []
            used_texts = set()
            
            # Strategy 1: Semantic similarity with adjusted threshold from settings
            for item in similarities[:top_k]:
                if item["score"] > settings.similarity_threshold:  # Use threshold from config
                    if item["text"] not in used_texts:
                        relevant_chunks.append(item["text"])
                        used_texts.add(item["text"])
            
            # Strategy 2: Comprehensive keyword matching
            query_lower = query.lower()
            
            # Define comprehensive keyword sets for different query types
            grace_period_keywords = [
                "grace", "grace period", "premium payment", "due date", "renewal", 
                "continue", "continuity", "thirty days", "30 days", "payment", 
                "premium", "renew", "lapse", "reinstate"
            ]
            
            waiting_period_keywords = [
                "waiting", "waiting period", "pre-existing", "pre existing", 
                "coverage", "months", "36 months", "thirty-six", "prior", 
                "effective date", "condition", "ailment", "disease"
            ]
            
            # Determine which keyword set to use
            keywords = []
            if any(term in query_lower for term in ["grace", "premium", "payment", "due"]):
                keywords.extend(grace_period_keywords)
            if any(term in query_lower for term in ["waiting", "pre-existing", "disease"]):
                keywords.extend(waiting_period_keywords)
            
            # If no specific keywords detected, use both sets
            if not keywords:
                keywords = grace_period_keywords + waiting_period_keywords
            
            # Search for keyword matches
            for item in similarities:
                text_lower = item["text"].lower()
                keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)
                
                # Include chunks with any keyword match
                if keyword_matches >= 1 and item["text"] not in used_texts:
                    relevant_chunks.append(item["text"])
                    used_texts.add(item["text"])
                    if len(relevant_chunks) >= 25:  # Generous limit
                        break
            
            # Strategy 3: Numerical pattern matching (for specific timeframes)
            number_patterns = ["30", "thirty", "36", "thirty-six", "days", "months"]
            for item in similarities:
                text_lower = item["text"].lower()
                if any(pattern in text_lower for pattern in number_patterns):
                    if item["text"] not in used_texts:
                        relevant_chunks.append(item["text"])
                        used_texts.add(item["text"])
                        if len(relevant_chunks) >= 30:
                            break
            
            # Strategy 4: Ensure minimum context (take top chunks regardless)
            if len(relevant_chunks) < 10:
                for item in similarities[:15]:  # Take top 15 regardless of score
                    if item["text"] not in used_texts:
                        relevant_chunks.append(item["text"])
                        used_texts.add(item["text"])
                        if len(relevant_chunks) >= 12:
                            break
            
            logger.info(f"Found {len(relevant_chunks)} relevant chunks for query: {query}")
            if similarities:
                logger.info(f"Top similarity score: {similarities[0]['score']:.3f}")
            
            # Log keyword matches for debugging
            keyword_chunk_count = sum(1 for chunk in relevant_chunks 
                                    if any(kw in chunk.lower() for kw in keywords[:5]))
            logger.info(f"Chunks with keyword matches: {keyword_chunk_count}")
            
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {str(e)}")
            return []

    def add_texts(self, texts: List[str], metadata: List[Dict] = None, document_id: str = None) -> List[str]:
        """Add texts to the vector store"""
        if metadata is None:
            metadata = [{} for _ in texts]
        
        # Generate embeddings using the configured model
        embeddings = self._get_embeddings(texts)
        
        # Generate IDs and store vectors
        vector_ids = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            vector_id = hashlib.md5(f"{text}_{i}".encode()).hexdigest()
            self.vectors[vector_id] = {
                "embedding": embedding,
                "metadata": metadata[i],
                "text": text
            }
            vector_ids.append(vector_id)
        
        # Associate with document if provided
        if document_id:
            if document_id not in self.documents:
                self.documents[document_id] = []
            self.documents[document_id].extend(vector_ids)
        
        return vector_ids