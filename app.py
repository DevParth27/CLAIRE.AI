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
from services.vector_store_lite import LightweightVectorStore
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

# Import the lightweight vector store
from services.vector_store_lite import LightweightVectorStore

# Initialize services with memory optimization
pdf_processor = PDFProcessor()
vector_store = LightweightVectorStore()  # Use lightweight version
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

# Timeout settings - Optimized for Render
API_TIMEOUT = 18  # Reduced from 25 to stay under Render's 30s limit
PER_QUESTION_TIMEOUT = 8  # Increased from 2 for better success rate
PDF_TIMEOUT = 6   # Reduced from 8
VECTOR_TIMEOUT = 3  # Reduced from 5

# ULTRA-AGGRESSIVE timeout settings
API_TIMEOUT = 12  # Reduced to 12 seconds total
PER_QUESTION_TIMEOUT = 4  # 4 seconds per question
PDF_TIMEOUT = 3   # 3 seconds for PDF
VECTOR_TIMEOUT = 2  # 2 seconds for vector operations

@app.post("/hackrx/run", response_model=QuestionResponse)
async def process_questions(
    request: QuestionRequest,
    token: str = Depends(verify_token)
):
    start_time = datetime.now()
    
    try:
        logger.info(f"Ultra-fast processing {len(request.questions)} questions")
        
        # Step 1: PDF processing (3s timeout)
        pdf_content = await asyncio.wait_for(
            pdf_processor.process_pdf_from_url(str(request.documents)),
            timeout=PDF_TIMEOUT
        )
        
        # Step 2: Vector storage (2s timeout)
        document_id = await asyncio.wait_for(
            vector_store.store_document(pdf_content, str(request.documents)),
            timeout=VECTOR_TIMEOUT
        )
        
        # Step 3: Process questions with strict limits
        answers = []
        for question in request.questions:
            try:
                # Check remaining time
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= API_TIMEOUT - 2:
                    answers.append("Timeout")
                    continue
                
                # Ultra-fast question processing
                chunks = await asyncio.wait_for(
                    vector_store.search_similar(question, document_id),
                    timeout=1.0  # 1 second for search
                )
                
                answer = await asyncio.wait_for(
                    qa_engine.generate_answer(question, chunks),
                    timeout=PER_QUESTION_TIMEOUT
                )
                
                answers.append(answer)
                
            except asyncio.TimeoutError:
                answers.append("Timeout")
            except Exception:
                answers.append("Error")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QuestionResponse(
            answers=answers,
            processing_time=processing_time,
            timeout_occurred=any("Timeout" in ans for ans in answers)
        )
        
    except asyncio.TimeoutError:
        return QuestionResponse(
            answers=["Processing timeout"] * len(request.questions),
            processing_time=(datetime.now() - start_time).total_seconds(),
            timeout_occurred=True
        )
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail="Processing failed")

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