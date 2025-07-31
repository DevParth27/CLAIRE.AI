import openai
import logging
import asyncio
from typing import List
from config_render import settings

logger = logging.getLogger(__name__)

class QAEngineRender:
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        logger.info("QA Engine Render initialized for ultra-fast processing")

    async def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        """Ultra-fast answer generation optimized for Render"""
        try:
            # Use minimal context for maximum speed
            context = context_chunks[0][:600] if context_chunks else "No context"
            
            # Ultra-minimal prompt for Render
            prompt = f"Context: {context}\n\nQ: {question}\nA (3 words max):"
            
            # API call with aggressive timeout
            async def make_api_call():
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,  # Very short responses
                    temperature=0,
                    timeout=2  # 2-second timeout for Render
                )
                return response.choices[0].message.content.strip()
            
            # Execute with strict timeout
            answer = await asyncio.wait_for(make_api_call(), timeout=3.0)
            return answer[:30] if answer else "Not found"
            
        except asyncio.TimeoutError:
            return "Timeout"
        except Exception as e:
            logger.error(f"Render QA Error: {e}")
            return "Error"