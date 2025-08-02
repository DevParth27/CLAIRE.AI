import aiohttp
import PyPDF2
import io
import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self):
        self.max_chunk_size = 800  # Reduced for better coherence
        self.chunk_overlap = 150   # Increased overlap
        self.min_chunk_size = 100  # Minimum viable chunk size
    
    async def process_pdf_from_url(self, pdf_url: str) -> List[Dict[str, str]]:
        """Download PDF from URL and extract text content"""
        try:
            # Download PDF
            async with aiohttp.ClientSession() as session:
                async with session.get(pdf_url) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to download PDF: HTTP {response.status}")
                    
                    pdf_content = await response.read()
            
            # Extract text with better formatting
            text_content = await self._extract_text_from_pdf(pdf_content)
            
            # Smart chunking with context preservation
            chunks = self._smart_chunk_text(text_content)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF from URL {pdf_url}: {str(e)}")
            raise
    
    async def _extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF content with better formatting"""
        text = ""
        
        try:
            with io.BytesIO(pdf_content) as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        # Clean and format text
                        cleaned_text = self._clean_text(page_text)
                        text += f"\n\n--- Page {page_num + 1} ---\n{cleaned_text}"
                        
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {str(e)}")
            raise Exception("Failed to extract text from PDF")
        
        if not text.strip():
            raise Exception("No text content found in PDF")
            
        return text
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Add space after punctuation
        # Preserve important formatting
        text = re.sub(r'\b(SECTION|CLAUSE|ARTICLE|CHAPTER)\b', r'\n\n\1', text, flags=re.IGNORECASE)
        return text.strip()
    
    def _smart_chunk_text(self, text: str) -> List[Dict[str, str]]:
        """Smart chunking that preserves context and meaning"""
        chunks = []
        
        # Split by major sections first
        sections = re.split(r'\n\n(?=(?:SECTION|CLAUSE|ARTICLE|CHAPTER|\d+\.))', text, flags=re.IGNORECASE)
        
        for section_idx, section in enumerate(sections):
            if len(section.strip()) < self.min_chunk_size:
                continue
                
            # Further split long sections by sentences
            sentences = re.split(r'(?<=[.!?])\s+', section)
            
            current_chunk = ""
            current_word_count = 0
            
            for sentence in sentences:
                sentence_words = len(sentence.split())
                
                # If adding this sentence would exceed chunk size, save current chunk
                if current_word_count + sentence_words > self.max_chunk_size and current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "chunk_id": len(chunks),
                        "section_id": section_idx,
                        "word_count": current_word_count
                    })
                    
                    # Start new chunk with overlap
                    overlap_sentences = current_chunk.split('. ')[-2:]  # Last 2 sentences
                    current_chunk = '. '.join(overlap_sentences) + '. ' + sentence
                    current_word_count = len(current_chunk.split())
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
                    current_word_count += sentence_words
            
            # Add remaining chunk
            if current_chunk.strip() and len(current_chunk.split()) >= self.min_chunk_size:
                chunks.append({
                    "text": current_chunk.strip(),
                    "chunk_id": len(chunks),
                    "section_id": section_idx,
                    "word_count": current_word_count
                })
        
        logger.info(f"Created {len(chunks)} smart chunks")
        return chunks