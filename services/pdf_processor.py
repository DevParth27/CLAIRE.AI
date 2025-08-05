import aiohttp
import PyPDF2
import io
import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self):
        self.max_chunk_size = 600  # Optimized for better coherence
        self.chunk_overlap = 100   # Increased overlap for context preservation
        self.min_chunk_size = 80   # Minimum viable chunk size
    
    async def process_pdf_from_url(self, pdf_url: str) -> List[Dict[str, str]]:
        """Download PDF from URL and extract text content with enhanced processing"""
        try:
            # Download PDF
            async with aiohttp.ClientSession() as session:
                async with session.get(pdf_url) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to download PDF: HTTP {response.status}")
                    
                    pdf_content = await response.read()
            
            # Extract text with enhanced formatting
            text_content = await self._extract_text_from_pdf(pdf_content)
            
            # Smart chunking with context preservation
            chunks = self._smart_chunk_text(text_content)
            
            logger.info(f"Successfully processed PDF with {len(chunks)} optimized chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF from URL {pdf_url}: {str(e)}")
            raise
    
    async def _extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF content with enhanced formatting preservation"""
        text = ""
        
        try:
            with io.BytesIO(pdf_content) as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        # Clean and format text while preserving structure
                        cleaned_text = self._clean_text(page_text)
                        text += f"\n\n=== PAGE {page_num + 1} ===\n{cleaned_text}"
                        
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {str(e)}")
            raise Exception("Failed to extract text from PDF")
        
        if not text.strip():
            raise Exception("No text content found in PDF")
            
        return text
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text while preserving important structure"""
        # Remove excessive whitespace but preserve paragraph breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Add space after punctuation
        text = re.sub(r'(\d)([A-Z])', r'\1 \2', text)     # Add space between number and letter
        
        # Preserve and enhance important policy structure markers
        text = re.sub(r'\b(SECTION|CLAUSE|ARTICLE|CHAPTER|PART)\s*(\d+)', r'\n\n\1 \2', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(\d+\.\d*|\d+)\s*(WAITING PERIOD|GRACE PERIOD|COVERAGE|BENEFIT|EXCLUSION|LIMIT)', r'\n\1 \2', text, flags=re.IGNORECASE)
        
        # Normalize insurance-specific terms for better matching
        text = re.sub(r'\bpre-existing\s+disease', 'pre-existing disease', text, flags=re.IGNORECASE)
        text = re.sub(r'\bno\s+claim\s+discount', 'no claim discount', text, flags=re.IGNORECASE)
        text = re.sub(r'\bwaiting\s+period', 'waiting period', text, flags=re.IGNORECASE)
        text = re.sub(r'\bgrace\s+period', 'grace period', text, flags=re.IGNORECASE)
        text = re.sub(r'\bpremium\s+payment', 'premium payment', text, flags=re.IGNORECASE)
        text = re.sub(r'\bmaternity\s+benefit', 'maternity benefit', text, flags=re.IGNORECASE)
        text = re.sub(r'\broom\s+rent', 'room rent', text, flags=re.IGNORECASE)
        text = re.sub(r'\bICU\s+charges', 'ICU charges', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _smart_chunk_text(self, text: str) -> List[Dict[str, str]]:
        """Advanced chunking that preserves context and policy structure"""
        chunks = []
        
        # First, split by major sections (pages, sections, clauses)
        major_sections = re.split(r'\n\n(?==== PAGE|SECTION|CLAUSE|ARTICLE)', text)
        
        for section_idx, section in enumerate(major_sections):
            if len(section.strip()) < self.min_chunk_size:
                continue
            
            # Check if section is small enough to be a single chunk
            section_words = section.split()
            if len(section_words) <= self.max_chunk_size:
                chunks.append({
                    "text": section.strip(),
                    "chunk_id": len(chunks),
                    "section_id": section_idx,
                    "word_count": len(section_words),
                    "chunk_type": "complete_section"
                })
                continue
            
            # For larger sections, split by sentences while preserving context
            sentences = self._split_into_sentences(section)
            
            current_chunk = ""
            current_word_count = 0
            sentence_buffer = []
            
            for sentence in sentences:
                sentence_words = len(sentence.split())
                
                # If adding this sentence would exceed chunk size
                if current_word_count + sentence_words > self.max_chunk_size and current_chunk:
                    # Save current chunk
                    chunks.append({
                        "text": current_chunk.strip(),
                        "chunk_id": len(chunks),
                        "section_id": section_idx,
                        "word_count": current_word_count,
                        "chunk_type": "split_section"
                    })
                    
                    # Create overlap for next chunk
                    overlap_sentences = sentence_buffer[-2:] if len(sentence_buffer) >= 2 else sentence_buffer
                    current_chunk = ' '.join(overlap_sentences) + ' ' + sentence if overlap_sentences else sentence
                    current_word_count = len(current_chunk.split())
                    sentence_buffer = overlap_sentences + [sentence]
                else:
                    current_chunk += ' ' + sentence if current_chunk else sentence
                    current_word_count += sentence_words
                    sentence_buffer.append(sentence)
                    
                    # Keep buffer manageable
                    if len(sentence_buffer) > 5:
                        sentence_buffer = sentence_buffer[-3:]
            
            # Add remaining chunk if substantial
            if current_chunk.strip() and len(current_chunk.split()) >= self.min_chunk_size:
                chunks.append({
                    "text": current_chunk.strip(),
                    "chunk_id": len(chunks),
                    "section_id": section_idx,
                    "word_count": current_word_count,
                    "chunk_type": "final_section"
                })
        
        logger.info(f"Created {len(chunks)} smart chunks with preserved context")
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while handling policy document specifics"""
        # Handle common abbreviations that shouldn't end sentences
        text = re.sub(r'\b(Mr|Mrs|Dr|Prof|Inc|Ltd|Co|etc|vs|i\.e|e\.g)\.', r'\1<DOT>', text)
        
        # Split on sentence endings
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore dots in abbreviations
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        
        # Filter out very short sentences (likely formatting artifacts)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return sentences