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
    
    async def search_similar(self, query: str, document_id: str, top_k: int = 5) -> List[str]:
        """Search for similar chunks in the in-memory vector database"""
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
                        "metadata": vector_data["metadata"]
                    })
            
            # Sort by similarity and get top_k
            similarities.sort(key=lambda x: x["score"], reverse=True)
            
            # Extract relevant text chunks
            relevant_chunks = []
            for item in similarities[:top_k]:
                if item["score"] > 0.3:  # Lower similarity threshold
                    relevant_chunks.append(item["metadata"]["text"])
            
            # Add debug logging
            logger.info(f"Found {len(relevant_chunks)} relevant chunks for query: {query}")
            if similarities:
                logger.info(f"Top similarity score: {similarities[0]['score']:.3f}")
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {str(e)}")
            return []