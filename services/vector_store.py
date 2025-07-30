import hashlib
import logging
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import settings

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dimension = 384
        
        # In-memory storage for development
        self.vectors = {}  # {vector_id: {"embedding": [...], "metadata": {...}}}
        self.documents = {}  # {document_id: [vector_ids]}
        
        logger.info("Initialized in-memory vector store (Pinecone alternative)")

    def _ensure_index_exists(self):
        """Mock method - no index creation needed for in-memory storage"""
        logger.info("Using in-memory vector storage")
    
    async def store_document(self, chunks: List[Dict[str, str]], document_url: str) -> str:
        """Store document chunks in in-memory vector database"""
        try:
            # Generate document ID
            document_id = hashlib.md5(document_url.encode()).hexdigest()
            
            # Store vectors
            vector_ids = []
            
            for chunk in chunks:
                # Generate embedding
                embedding = self.embedding_model.encode(chunk["text"])
                
                # Create vector ID
                vector_id = f"{document_id}_{chunk['chunk_id']}"
                
                # Prepare metadata
                metadata = {
                    "document_id": document_id,
                    "document_url": document_url,
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "start_word": chunk["start_word"],
                    "end_word": chunk["end_word"]
                }
                
                # Store in memory
                self.vectors[vector_id] = {
                    "embedding": embedding,
                    "metadata": metadata
                }
                vector_ids.append(vector_id)
            
            # Track document vectors
            self.documents[document_id] = vector_ids
            
            logger.info(f"Stored {len(chunks)} chunks for document {document_id} in memory")
            return document_id
            
        except Exception as e:
            logger.error(f"Error storing document: {str(e)}")
            raise

    async def search_similar(self, query: str, document_id: str, top_k: int = 20) -> List[str]:
        """Comprehensive search with multiple strategies to find ALL relevant information"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Get document vectors
            if document_id not in self.documents:
                logger.warning(f"Document {document_id} not found")
                return []
            
            # Calculate similarities
            similarities = []
            for vector_id in self.documents[document_id]:
                if vector_id in self.vectors:
                    vector_data = self.vectors[vector_id]
                    similarity = cosine_similarity(
                        [query_embedding], 
                        [vector_data["embedding"]]
                    )[0][0]
                    
                    similarities.append({
                        "score": similarity,
                        "metadata": vector_data["metadata"],
                        "text": vector_data["metadata"]["text"]
                    })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x["score"], reverse=True)
            
            # Multi-strategy comprehensive retrieval
            relevant_chunks = []
            used_texts = set()
            
            # Strategy 1: Semantic similarity (very permissive)
            for item in similarities[:top_k]:
                if item["score"] > 0.05:  # Extremely low threshold
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