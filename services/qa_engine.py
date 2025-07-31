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
        return len(text.split()) * 1.3  # Rough estimation

    def organize_context(self, chunks: List[str], max_tokens: int = 3000) -> str:
        """Organize context chunks within token limit"""
        context_parts = []
        current_tokens = 0
        
        for chunk in chunks[:5]:  # Use top 5 chunks
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
            
            # Create comprehensive prompt
            system_prompt = """
You are an expert AI assistant that analyzes policy documents and provides accurate, detailed answers.

Instructions:
1. Provide comprehensive and accurate answers based on the given context
2. If the information is not in the context, clearly state that
3. Use specific details and quotes from the context when relevant
4. Structure your answer clearly with bullet points or paragraphs as appropriate
5. Be concise but thorough
"""
            
            user_prompt = f"""
Context from document:
{context}

Question: {question}

Please provide a detailed answer based on the context above.
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
                        temperature=0.1,
                        timeout=30
                    )
                    
                    answer = response.choices[0].message.content.strip()
                    return self._format_answer(answer)
                    
                except Exception as e:
                    logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                    if attempt == self.max_retries - 1:
                        raise e
                    await asyncio.sleep(1)  # Brief delay before retry
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I encountered an error while processing your question. Please try again."
    
    def _format_answer(self, answer: str) -> str:
        """Format and clean up the generated answer"""
        if not answer:
            return "I couldn't generate an answer for your question."
        
        # Clean up the answer
        answer = answer.strip()
        
        # Ensure it's not too long
        if len(answer) > 1000:
            answer = answer[:997] + "..."
        
        return answer
