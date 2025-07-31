import openai
import logging
import asyncio
from typing import List
from config import settings

logger = logging.getLogger(__name__)

class QAEngine:
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        logger.info("QA Engine initialized for ultra-fast processing")

    async def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        """Ultra-fast answer generation with aggressive timeouts"""
        try:
            # Use only the first chunk for maximum speed
            context = context_chunks[0][:800] if context_chunks else "No context"
            
            # Ultra-minimal prompt for speed
            prompt = f"Context: {context}\n\nQ: {question}\nA (5 words max):"
            
            # Create the API call with timeout
            async def make_api_call():
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",  # Faster than GPT-4
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=15,
                    temperature=0,
                    timeout=3  # 3-second timeout
                )
                return response.choices[0].message.content.strip()
            
            # Execute with asyncio timeout
            answer = await asyncio.wait_for(make_api_call(), timeout=4.0)
            return answer[:50] if answer else "Not found"
            
        except asyncio.TimeoutError:
            return "Timeout - info not found"
        except Exception as e:
            logger.error(f"QA Error: {e}")
            return "Error processing"
