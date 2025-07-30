from sqlalchemy import Column, Integer, String, DateTime, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from config import settings

Base = declarative_base()

class DocumentProcessing(Base):
    __tablename__ = "document_processing"
    
    id = Column(Integer, primary_key=True, index=True)
    document_url = Column(String(500), nullable=False)  # Added length for SQLite
    document_id = Column(String(100), nullable=False, index=True)  # Added length
    processing_time = Column(Integer)  # in seconds
    questions_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

class QuestionAnswer(Base):
    __tablename__ = "question_answers"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(String(100), nullable=False, index=True)  # Added length
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    processing_time = Column(Integer)  # in milliseconds
    created_at = Column(DateTime, default=datetime.utcnow)

# Database setup - Updated for SQLite
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

async def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)