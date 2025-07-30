from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List
import os
import asyncio
import aiohttp
import logging
from datetime import datetime

# Import our custom modules
from services.pdf_processor import PDFProcessor
from services.vector_store import VectorStore
from services.qa_engine import QAEngine
from database.models import init_db
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HackRx AI QA System",
    description="AI-powered question-answering system for policy documents",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Pydantic models
class QuestionRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class QuestionResponse(BaseModel):
    answers: List[str]

# Initialize services
pdf_processor = PDFProcessor()
vector_store = VectorStore()
qa_engine = QAEngine()

# Authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    expected_token = "433c9562217435ac71d779508405bfa9b20d0f58ff2aeb482c16c0e251f9f85f"
    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

@app.on_event("startup")
async def startup_event():
    """Initialize database and services on startup"""
    await init_db()
    logger.info("Application started successfully")

@app.get("/")
async def root():
    return {"message": "HackRx AI QA System is running", "timestamp": datetime.now()}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

@app.post("/hackrx/run", response_model=QuestionResponse)
async def process_questions(
    request: QuestionRequest,
    token: str = Depends(verify_token)
):
    """Main endpoint for processing questions against policy documents"""
    try:
        start_time = datetime.now()
        logger.info(f"Processing request with {len(request.questions)} questions")
        
        # Step 1: Download and process PDF
        pdf_content = await pdf_processor.process_pdf_from_url(str(request.documents))
        
        # Step 2: Create embeddings and store in vector database
        document_id = await vector_store.store_document(pdf_content, str(request.documents))
        
        # Step 3: Process each question
        answers = []
        for question in request.questions:
            try:
                # Retrieve relevant context
                relevant_chunks = await vector_store.search_similar(question, document_id)
                
                # Generate answer using Gemini
                answer = await qa_engine.generate_answer(question, relevant_chunks)
                answers.append(answer)
                
            except Exception as e:
                logger.error(f"Error processing question '{question}': {str(e)}")
                answers.append("Information not available in the document.")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Request processed in {processing_time:.2f} seconds")
        
        return QuestionResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while processing request"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)