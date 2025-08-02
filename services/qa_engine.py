import openai
import logging
import asyncio
from typing import List, Dict
from config import settings
import re

logger = logging.getLogger(__name__)

class QAEngine:
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.max_retries = 3
        logger.info("Enhanced QA Engine initialized successfully")

    def estimate_tokens(self, text: str) -> int:
        """More accurate token estimation"""
        return len(text.split()) * 1.4  # Slightly higher estimate

    def organize_context(self, chunks: List[str], max_tokens: int = 6000) -> str:
        """Intelligently organize context chunks"""
        if not chunks:
            return ""
        
        # Deduplicate similar chunks
        unique_chunks = self._deduplicate_chunks(chunks)
        
        # Prioritize chunks with policy-specific keywords
        prioritized_chunks = self._prioritize_chunks(unique_chunks)
        
        context_parts = []
        current_tokens = 0
        
        for i, chunk in enumerate(prioritized_chunks[:10]):  # Max 10 chunks
            chunk_tokens = self.estimate_tokens(chunk)
            if current_tokens + chunk_tokens > max_tokens:
                break
            
            # Add section markers for clarity
            formatted_chunk = f"\n--- Context Section {i+1} ---\n{chunk}"
            context_parts.append(formatted_chunk)
            current_tokens += chunk_tokens
        
        return "\n".join(context_parts)
    
    def _deduplicate_chunks(self, chunks: List[str]) -> List[str]:
        """Remove highly similar chunks"""
        unique_chunks = []
        
        for chunk in chunks:
            is_duplicate = False
            chunk_words = set(chunk.lower().split())
            
            for existing_chunk in unique_chunks:
                existing_words = set(existing_chunk.lower().split())
                overlap = len(chunk_words.intersection(existing_words))
                similarity = overlap / max(len(chunk_words), len(existing_words))
                
                if similarity > 0.8:  # 80% similarity threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _prioritize_chunks(self, chunks: List[str]) -> List[str]:
        """Prioritize chunks based on relevance indicators"""
        priority_keywords = [
            'waiting period', 'grace period', 'coverage', 'benefit', 'premium',
            'claim', 'policy', 'maternity', 'pre-existing', 'discount',
            'hospital', 'treatment', 'ayush', 'room rent', 'icu'
        ]
        
        scored_chunks = []
        for chunk in chunks:
            chunk_lower = chunk.lower()
            score = sum(1 for keyword in priority_keywords if keyword in chunk_lower)
            
            # Boost score for chunks with specific numbers/percentages
            if re.search(r'\d+\s*(days?|months?|years?|%)', chunk_lower):
                score += 2
            
            scored_chunks.append((chunk, score))
        
        # Sort by score (descending) and return chunks
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in scored_chunks]

    async def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        """Generate comprehensive answer with enhanced prompting"""
        try:
            # Organize context intelligently
            context = self.organize_context(context_chunks)
            
            if not context:
                return "I couldn't find relevant information in the document to answer your question."
            
            # Enhanced system prompt for better accuracy
            system_prompt = """
You are an expert insurance policy analyst with deep knowledge of policy documents and insurance terminology.

CRITICAL INSTRUCTIONS:
1. ONLY use information explicitly stated in the provided context sections
2. For specific details (waiting periods, coverage limits, percentages, amounts), quote EXACT values from the document
3. If information is not in the context, clearly state "This information is not mentioned in the provided document sections"
4. For yes/no questions, provide definitive answers based on the context
5. Include specific policy terms, conditions, and numerical values when available
6. Structure answers clearly with bullet points for multiple details
7. Be precise about coverage inclusions and exclusions
8. When mentioning waiting periods, grace periods, or time limits, always include the exact duration
9. For coverage questions, specify what is covered and any conditions or limitations
10. If multiple context sections contain relevant information, synthesize them coherently
"""
            
            # Enhanced user prompt with question analysis
            question_type = self._analyze_question_type(question)
            
            user_prompt = f"""
Policy Document Context Sections:
{context}

Question Type: {question_type}
Question: {question}

Provide a precise, comprehensive answer based ONLY on the information in the context sections above. 

For factual questions (waiting periods, amounts, percentages): Quote exact values.
For coverage questions: Specify what is covered, conditions, and limitations.
For yes/no questions: Provide clear yes/no followed by supporting details.

Answer:
"""
            
            # Make API call with retries
            for attempt in range(self.max_retries):
                try:
                    response = await asyncio.to_thread(
                        self.client.chat.completions.create,
                        model=settings.openai_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=settings.max_tokens,
                        temperature=0.0,  # Zero temperature for consistency
                        timeout=30
                    )
                    
                    answer = response.choices[0].message.content.strip()
                    return self._format_answer(answer)
                    
                except Exception as e:
                    logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                    if attempt == self.max_retries - 1:
                        raise e
                    await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I encountered an error while processing your question. Please try again."
    
    def _analyze_question_type(self, question: str) -> str:
        """Analyze question type for better prompting"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['what is', 'how much', 'how long']):
            return "Factual/Specific Detail"
        elif any(word in question_lower for word in ['does', 'is there', 'are there', 'can']):
            return "Yes/No Coverage"
        elif 'waiting period' in question_lower:
            return "Waiting Period"
        elif any(word in question_lower for word in ['cover', 'coverage', 'benefit']):
            return "Coverage/Benefits"
        elif any(word in question_lower for word in ['how', 'define', 'definition']):
            return "Definition/Process"
        else:
            return "General Information"
    
    def _format_answer(self, answer: str) -> str:
        """Format and clean up the generated answer"""
        if not answer:
            return "I couldn't generate an answer for your question."
        
        answer = answer.strip()
        
        # Allow longer answers for detailed policy information
        if len(answer) > 2000:
            answer = answer[:1997] + "..."
        
        # Ensure proper formatting
        if not answer.endswith(('.', '!', '?')):
            answer += "."
        
        return answer