from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from services.qa_engine import QAEngine
from services.qa_engine_enhanced import EnhancedQAEngine
from typing import List
import os
import asyncio
import logging
from datetime import datetime
import json
import uuid
from dotenv import load_dotenv
# Import our custom modules
from services.pdf_processor import PDFProcessor
from services.vector_store_lite import LightweightVectorStore
from services.qa_engine import QAEngine
from database.models import init_db
from config import settings
load_dotenv()
# Configure logging
# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hackrx_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
execution_logger = logging.getLogger('execution_tracker')

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
vector_store = LightweightVectorStore()
#qa_engine = QAEngine()
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

# Cost-optimized timeout settings
API_TIMEOUT = 240   # 4 minutes total for larger batches
PER_QUESTION_TIMEOUT = 60  # Increased from 40 seconds
PDF_TIMEOUT = 20   # Increased from 15 to 20 seconds
VECTOR_TIMEOUT = 10  # Increased from 5 to 10 seconds

@app.post("/api/v1/hackrx/run", response_model=QuestionResponse)
async def process_questions(
    request: QuestionRequest,
    token: str = Depends(verify_token)
):
    # Generate unique session ID for tracking
    session_id = str(uuid.uuid4())[:8]
    start_time = datetime.now()
    
    # Log API request initiation
    execution_logger.info(f"SESSION_START|{session_id}|{start_time.isoformat()}|Questions:{len(request.questions)}|Document:{str(request.documents)[-50:]}")
    
    try:
        # Support larger batch sizes
        if len(request.questions) > 50:
            execution_logger.warning(f"SESSION_ERROR|{session_id}|Too many questions: {len(request.questions)}")
            raise HTTPException(
                status_code=400, 
                detail="Maximum 50 questions per batch for cost optimization"
            )
        
        # Log each question before processing
        for i, question in enumerate(request.questions, 1):
            execution_logger.info(f"QUESTION_RECEIVED|{session_id}|Q{i}|{question[:100]}{'...' if len(question) > 100 else ''}")
        
        logger.info(f"Processing {len(request.questions)} questions in optimized batch mode")
        
        # Step 1: PDF processing
        pdf_start = datetime.now()
        try:
            pdf_content = await asyncio.wait_for(
                pdf_processor.process_pdf_from_url(str(request.documents)),
                timeout=PDF_TIMEOUT
            )
            pdf_time = (datetime.now() - pdf_start).total_seconds()
            execution_logger.info(f"PDF_PROCESSED|{session_id}|Time:{pdf_time:.2f}s|Content_length:{len(pdf_content)}")
        except asyncio.TimeoutError:
            execution_logger.error(f"PDF_TIMEOUT|{session_id}|{PDF_TIMEOUT}s")
            logger.error("PDF processing timeout")
            raise HTTPException(status_code=408, detail="PDF processing timeout")
        
        # Step 2: Vector storage
        vector_start = datetime.now()
        try:
            document_id = await asyncio.wait_for(
                vector_store.store_document(pdf_content, str(request.documents)),
                timeout=VECTOR_TIMEOUT
            )
            vector_time = (datetime.now() - vector_start).total_seconds()
            execution_logger.info(f"VECTOR_STORED|{session_id}|Time:{vector_time:.2f}s|Doc_ID:{document_id}")
        except asyncio.TimeoutError:
            execution_logger.error(f"VECTOR_TIMEOUT|{session_id}|{VECTOR_TIMEOUT}s")
            logger.error("Vector storage timeout")
            raise HTTPException(status_code=408, detail="Vector storage timeout")
        
        # Step 3: Process questions in parallel batches
        answers = [None] * len(request.questions)  # Pre-allocate answers list
        
        # Process questions in batches with parallel execution
        async def process_question(idx, question):
            question_start = datetime.now()
            execution_logger.info(f"QUESTION_START|{session_id}|Q{idx+1}|{question}")
            
            try:
                # Get optimal number of chunks
                chunks_start = datetime.now()
                chunks = await asyncio.wait_for(
                    vector_store.search_similar(question, document_id, top_k=settings.vector_top_k),
                    timeout=5.0
                )
                chunks_time = (datetime.now() - chunks_start).total_seconds()
                execution_logger.info(f"CHUNKS_RETRIEVED|{session_id}|Q{idx+1}|Time:{chunks_time:.2f}s|Count:{len(chunks)}")
                
                # Generate answer
                answer_start = datetime.now()
                answer = await asyncio.wait_for(
                    qa_engine.generate_answer(question, chunks),
                    timeout=PER_QUESTION_TIMEOUT
                )
                answer_time = (datetime.now() - answer_start).total_seconds()
                question_total_time = (datetime.now() - question_start).total_seconds()
                
                answers[idx] = answer
                
                # Log successful answer generation
                execution_logger.info(f"ANSWER_GENERATED|{session_id}|Q{idx+1}|Time:{answer_time:.2f}s|Total:{question_total_time:.2f}s|Length:{len(answer)}")
                execution_logger.info(f"ANSWER_CONTENT|{session_id}|Q{idx+1}|{answer[:200]}{'...' if len(answer) > 200 else ''}")
                
            except asyncio.TimeoutError:
                timeout_answer = "Processing timeout - please try a simpler question"
                answers[idx] = timeout_answer
                execution_logger.error(f"QUESTION_TIMEOUT|{session_id}|Q{idx+1}|{PER_QUESTION_TIMEOUT}s")
            except Exception as e:
                error_answer = "Error processing question - please try again"
                answers[idx] = error_answer
                execution_logger.error(f"QUESTION_ERROR|{session_id}|Q{idx+1}|{str(e)}")
        
        # Process questions in batches with the configured batch size
        batch_size = settings.batch_size  # Use batch_size from config
        
        # Process questions in parallel with the configured number of parallel questions
        semaphore = asyncio.Semaphore(settings.parallel_questions)  # Use parallel_questions from config
        for i in range(0, len(request.questions), batch_size):
            batch = request.questions[i:i+batch_size]
            batch_tasks = [process_question(i+j, question) for j, question in enumerate(batch)]
            await asyncio.gather(*batch_tasks)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log session completion
        execution_logger.info(f"SESSION_COMPLETE|{session_id}|Total_time:{processing_time:.2f}s|Timeout_occurred:{any('timeout' in ans.lower() for ans in answers)}|Processing_time:{processing_time:.5f}|Success_rate:{len([a for a in answers if 'timeout' not in a.lower() and 'error' not in a.lower()])}/{len(answers)}")
        
        return QuestionResponse(
            answers=answers,
        )
        
    except Exception as e:
        execution_logger.error(f"SESSION_FAILED|{session_id}|{str(e)}")
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