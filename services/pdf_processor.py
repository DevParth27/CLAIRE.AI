import aiohttp
import aiofiles
import PyPDF2
# import pdfplumber  # Comment out or remove this line
import io
import logging
from typing import List, Dict
import tempfile
import os

logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self):
        self.max_chunk_size = 1000
        self.chunk_overlap = 200
    
    async def process_pdf_from_url(self, pdf_url: str) -> List[Dict[str, str]]:
        """Download PDF from URL and extract text content"""
        try:
            # Download PDF
            async with aiohttp.ClientSession() as session:
                async with session.get(pdf_url) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to download PDF: HTTP {response.status}")
                    
                    pdf_content = await response.read()
            
            # Extract text
            text_content = await self._extract_text_from_pdf(pdf_content)
            
            # Chunk the content
            chunks = self._chunk_text(text_content)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF from URL {pdf_url}: {str(e)}")
            raise
    
    async def _extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF content using PyPDF2 only"""
        text = ""
        
        try:
            # Use only PyPDF2 for memory efficiency
            with io.BytesIO(pdf_content) as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                        
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {str(e)}")
            raise Exception("Failed to extract text from PDF")
        
        if not text.strip():
            raise Exception("No text content found in PDF")
            
        return text
    
    def _chunk_text(self, text: str) -> List[Dict[str, str]]:
        """Split text into overlapping chunks"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.max_chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.max_chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "text": chunk_text,
                "chunk_id": len(chunks),
                "start_word": i,
                "end_word": i + len(chunk_words)
            })
        
        return chunks