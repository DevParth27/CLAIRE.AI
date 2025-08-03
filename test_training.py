import asyncio
import logging
import time
import random
from typing import List, Dict

from training_data import TrainingDataManager
from services.qa_engine import QAEngine
from services.vector_store_lite import LightweightVectorStore
from services.pdf_processor import PDFProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, pdf_url: str):
        self.pdf_url = pdf_url
        self.pdf_processor = PDFProcessor()
        self.vector_store = LightweightVectorStore()
        self.qa_engine = QAEngine()
        self.training_manager = TrainingDataManager()
        self.document_id = None
    
    async def initialize(self):
        """Initialize by processing the PDF and storing it in the vector store"""
        logger.info(f"Processing PDF from {self.pdf_url}")
        pdf_content = await self.pdf_processor.process_pdf_from_url(self.pdf_url)
        logger.info(f"Extracted {len(pdf_content)} chunks from PDF")
        
        self.document_id = await self.vector_store.store_document(pdf_content, self.pdf_url)
        logger.info(f"Stored document with ID: {self.document_id}")
    
    async def evaluate_category(self, category: str, sample_size: int = 5):
        """Evaluate model performance on a specific category of questions"""
        if not self.document_id:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        questions = self.training_manager.get_questions(category)
        if not questions:
            logger.warning(f"No questions found for category: {category}")
            return {}
        
        # Sample questions if needed
        if sample_size and sample_size < len(questions):
            sampled_questions = random.sample(questions, sample_size)
        else:
            sampled_questions = questions
        
        results = []
        total_time = 0
        
        for i, question in enumerate(sampled_questions, 1):
            logger.info(f"[{i}/{len(sampled_questions)}] Processing: {question}")
            
            start_time = time.time()
            
            # Get relevant chunks
            chunks = await self.vector_store.search_similar(question, self.document_id, top_k=8)
            logger.info(f"Found {len(chunks)} relevant chunks")
            
            # Generate answer
            answer = await self.qa_engine.generate_answer(question, chunks)
            
            elapsed = time.time() - start_time
            total_time += elapsed
            
            results.append({
                "question": question,
                "answer": answer,
                "chunks_count": len(chunks),
                "time_seconds": elapsed
            })
            
            logger.info(f"Generated answer in {elapsed:.2f} seconds")
            logger.info(f"Answer: {answer[:100]}..." if len(answer) > 100 else f"Answer: {answer}")
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(1)
        
        # Calculate statistics
        avg_time = total_time / len(results) if results else 0
        avg_chunks = sum(r["chunks_count"] for r in results) / len(results) if results else 0
        
        logger.info(f"Evaluation complete for {category}")
        logger.info(f"Average processing time: {avg_time:.2f} seconds")
        logger.info(f"Average chunks retrieved: {avg_chunks:.1f}")
        
        return {
            "category": category,
            "questions_count": len(sampled_questions),
            "avg_time": avg_time,
            "avg_chunks": avg_chunks,
            "results": results
        }
    
    async def evaluate_all_categories(self, sample_size: int = 3):
        """Evaluate all categories"""
        all_results = {}
        
        for category in self.training_manager.categories.keys():
            logger.info(f"Evaluating category: {category}")
            results = await self.evaluate_category(category, sample_size)
            all_results[category] = results
        
        return all_results

async def main():
    # Use the same PDF URL as in your test_api.py
    pdf_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    evaluator = ModelEvaluator(pdf_url)
    await evaluator.initialize()
    
    # Evaluate specific categories or all
    results = await evaluator.evaluate_category("policy_specific", sample_size=5)
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    asyncio.run(main())