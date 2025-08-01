import openai
import logging
import asyncio
from typing import List, Dict
from config import settings

logger = logging.getLogger(__name__)

class QAEngine:
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.max_retries = 3
        logger.info("QA Engine initialized successfully")

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        return len(text.split()) * 1.3

    def organize_context(self, chunks: List[str], max_tokens: int = 4000) -> str:
        """Organize context chunks within token limit"""
        context_parts = []
        current_tokens = 0
        
        # Use more chunks for better context
        for chunk in chunks[:8]:
            chunk_tokens = self.estimate_tokens(chunk)
            if current_tokens + chunk_tokens > max_tokens:
                break
            context_parts.append(chunk)
            current_tokens += chunk_tokens
        
        return "\n\n".join(context_parts)

    async def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        """Generate comprehensive answer using OpenAI API"""
        try:
            # Organize context
            context = self.organize_context(context_chunks)
            
            if not context:
                return "I couldn't find relevant information in the document to answer your question."
            
            # Enhanced system prompt for policy documents
            system_prompt = """
You are an expert insurance policy analyst. Analyze the provided policy document context and answer questions with precision.

CRITICAL INSTRUCTIONS:
1. ONLY use information explicitly stated in the provided context
2. For specific details (waiting periods, coverage limits, percentages), quote exact values from the document
3. If information is not in the context, clearly state "This information is not mentioned in the provided document"
4. For yes/no questions, provide definitive answers based on the context
5. Include specific policy terms, conditions, and numerical values when available
6. Structure answers clearly with bullet points for multiple details
7. Be precise about coverage inclusions and exclusions
"""
            
            user_prompt = f"""
Policy Document Context:
{context}

Question: {question}

Provide a precise answer based ONLY on the information in the context above. Include specific details, waiting periods, coverage limits, and conditions mentioned in the document.
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
    
    def _format_answer(self, answer: str) -> str:
        """Format and clean up the generated answer"""
        if not answer:
            return "I couldn't generate an answer for your question."
        
        answer = answer.strip()
        
        # Allow longer answers for detailed policy information
        if len(answer) > 1500:
            answer = answer[:1497] + "..."
        
        return answer
