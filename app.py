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
    processing_time: float
    timeout_occurred: bool = False

# Initialize services
pdf_processor = PDFProcessor()
vector_store = VectorStore()
qa_engine = QAEngine()

# Authentication - moved before endpoint definition
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    expected_token = "433c9562217435ac71d779508405bfa9b20d0f58ff2aeb482c16c0e251f9f85f"
    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Timeout settings
API_TIMEOUT = 25  # Reduced from 30
PER_QUESTION_TIMEOUT = 2  # Reduced from 5
PDF_TIMEOUT = 8  # Reduced from 15
VECTOR_TIMEOUT = 5  # Reduced from 10

@app.post("/hackrx/run", response_model=QuestionResponse)
async def process_questions(
    request: QuestionRequest,
    token: str = Depends(verify_token)
):
    start_time = datetime.now()
    timeout_occurred = False
    
    try:
        logger.info(f"Processing {len(request.questions)} questions with speed optimization")
        
        # Step 1: PDF processing with reduced timeout
        pdf_content = await process_with_timeout(
            pdf_processor.process_pdf_from_url(str(request.documents)), 
            timeout_seconds=PDF_TIMEOUT,
            fallback_result=None
        )
        
        if pdf_content is None:
            return QuestionResponse(
                answers=["PDF processing timed out."] * len(request.questions),
                processing_time=(datetime.now() - start_time).total_seconds(),
                timeout_occurred=True
            )
        
        # Step 2: Vector storage with reduced timeout
        document_id = await process_with_timeout(
            vector_store.store_document(pdf_content, str(request.documents)),
            timeout_seconds=VECTOR_TIMEOUT,
            fallback_result=None
        )
        
        if document_id is None:
            return QuestionResponse(
                answers=["Document indexing timed out."] * len(request.questions),
                processing_time=(datetime.now() - start_time).total_seconds(),
                timeout_occurred=True
            )
        
        # Step 3: Process questions with strict timeouts
        answers = []
        for question in request.questions:
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed >= API_TIMEOUT - 3:  # Leave 3 seconds buffer
                answers.append("Request timed out.")
                timeout_occurred = True
                continue
                
            try:
                async def process_single_question():
                    chunks = await vector_store.search_similar(question, document_id)
                    return await qa_engine.generate_answer(question, chunks)
                
                answer = await process_with_timeout(
                    process_single_question(),
                    timeout_seconds=PER_QUESTION_TIMEOUT,
                    fallback_result="Answer timed out."
                )
                
                answers.append(answer)
                
            except Exception as e:
                answers.append("Error processing question.")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QuestionResponse(
            answers=answers,
            processing_time=processing_time,
            timeout_occurred=timeout_occurred
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Processing error"
        )

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

async def process_with_timeout(coro, timeout_seconds: float, fallback_result=None):
    """Execute coroutine with timeout and return fallback on timeout"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout_seconds} seconds")
        return fallback_result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        timeout_keep_alive=35,  # Slightly longer than API timeout
        timeout_graceful_shutdown=5
    )