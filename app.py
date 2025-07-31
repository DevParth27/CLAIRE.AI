from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List
import os
import asyncio
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

# Initialize services
pdf_processor = PDFProcessor()
vector_store = LightweightVectorStore()
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

# Original timeout settings (more generous)
API_TIMEOUT = 30  # 30 seconds total
PER_QUESTION_TIMEOUT = 15  # 15 seconds per question
PDF_TIMEOUT = 10   # 10 seconds for PDF
VECTOR_TIMEOUT = 5  # 5 seconds for vector operations

@app.post("/api/v1/hackrx/run", response_model=QuestionResponse)
async def process_questions(
    request: QuestionRequest,
    token: str = Depends(verify_token)
):
    start_time = datetime.now()
    
    try:
        logger.info(f"Processing {len(request.questions)} questions")
        
        # Step 1: PDF processing
        try:
            pdf_content = await asyncio.wait_for(
                pdf_processor.process_pdf_from_url(str(request.documents)),
                timeout=PDF_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error("PDF processing timeout")
            raise HTTPException(status_code=408, detail="PDF processing timeout")
        
        # Step 2: Vector storage
        try:
            document_id = await asyncio.wait_for(
                vector_store.store_document(pdf_content, str(request.documents)),
                timeout=VECTOR_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error("Vector storage timeout")
            raise HTTPException(status_code=408, detail="Vector storage timeout")
        
        # Step 3: Process questions
        answers = []
        for question in request.questions:
            try:
                # Check remaining time
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= API_TIMEOUT - 5:
                    answers.append("Processing timeout - please try with fewer questions")
                    continue
                
                # Search for relevant chunks
                chunks = await asyncio.wait_for(
                    vector_store.search_similar(question, document_id),
                    timeout=3.0
                )
                
                # Generate answer
                answer = await asyncio.wait_for(
                    qa_engine.generate_answer(question, chunks),
                    timeout=PER_QUESTION_TIMEOUT
                )
                
                answers.append(answer)
                
            except asyncio.TimeoutError:
                answers.append("Question processing timeout - please try a simpler question")
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                answers.append("Error processing question - please try again")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QuestionResponse(
            answers=answers,
            processing_time=processing_time,
            timeout_occurred=any("timeout" in ans.lower() for ans in answers)
        )
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000)),
        timeout_keep_alive=60,
        timeout_graceful_shutdown=10
    )