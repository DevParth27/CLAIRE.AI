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
from config_railway import settings  # Changed from 'config' to 'config_railway'

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
    if credentials.credentials != settings.bearer_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Timeouts for Railway
API_TIMEOUT = 30
PER_QUESTION_TIMEOUT = 15
PDF_TIMEOUT = 10
VECTOR_TIMEOUT = 5

@app.post("/hackrx/run", response_model=QuestionResponse)
async def process_questions(
    request: QuestionRequest,
    token: str = Depends(verify_token)
):
    start_time = datetime.now()
    timeout_occurred = False
    
    try:
        # Process with timeout
        result = await asyncio.wait_for(
            _process_questions_internal(request),
            timeout=API_TIMEOUT
        )
        return result
    except asyncio.TimeoutError:
        timeout_occurred = True
        processing_time = (datetime.now() - start_time).total_seconds()
        return QuestionResponse(
            answers=["Request timed out. Please try with fewer questions or simpler documents."],
            processing_time=processing_time,
            timeout_occurred=True
        )
    except Exception as e:
        logger.error(f"Error processing questions: {str(e)}")
        processing_time = (datetime.now() - start_time).total_seconds()
        return QuestionResponse(
            answers=[f"Error: {str(e)}"],
            processing_time=processing_time,
            timeout_occurred=timeout_occurred
        )

async def _process_questions_internal(request: QuestionRequest):
    start_time = datetime.now()
    
    # Download and process PDF
    pdf_content = await asyncio.wait_for(
        pdf_processor.extract_text_from_url(str(request.documents)),
        timeout=PDF_TIMEOUT
    )
    
    # Create vector store
    await asyncio.wait_for(
        vector_store.create_embeddings(pdf_content),
        timeout=VECTOR_TIMEOUT
    )
    
    # Process questions
    answers = []
    for question in request.questions[:3]:  # Limit to 3 questions for Railway
        try:
            answer = await asyncio.wait_for(
                qa_engine.answer_question(question, vector_store),
                timeout=PER_QUESTION_TIMEOUT
            )
            answers.append(answer)
        except asyncio.TimeoutError:
            answers.append("Question processing timed out.")
        except Exception as e:
            answers.append(f"Error processing question: {str(e)}")
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    return QuestionResponse(
        answers=answers,
        processing_time=processing_time,
        timeout_occurred=False
    )

@app.on_event("startup")
async def startup_event():
    await init_db()
    logger.info("Application started successfully on Railway")

@app.get("/")
async def root():
    return {"message": "HackRx AI QA System is running on Railway!", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "platform": "railway"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        timeout_keep_alive=60,
        timeout_graceful_shutdown=10
    )