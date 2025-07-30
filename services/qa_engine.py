import openai
import logging
from typing import List
from config import settings

logger = logging.getLogger(__name__)

class QAEngine:
    def __init__(self):
        # Debug: Log the configuration (without exposing the full API key)
        logger.info(f"Initializing QA Engine with base_url: {settings.openai_base_url}")
        logger.info(f"API key starts with: {settings.openai_api_key[:10]}...")
        logger.info(f"Using model: {settings.openai_model}")
        
        # Configure OpenAI client for OpenRouter with error handling
        try:
            self.client = openai.OpenAI(
                base_url=settings.openai_base_url,
                api_key=settings.openai_api_key
            )
        except TypeError as e:
            # Fallback for older OpenAI versions or compatibility issues
            logger.warning(f"OpenAI client initialization failed with TypeError: {e}")
            logger.info("Attempting fallback initialization...")
            self.client = openai.OpenAI(
                api_key=settings.openai_api_key
            )
            # Set base_url manually if the client supports it
            if hasattr(self.client, '_base_url'):
                self.client._base_url = settings.openai_base_url
    
    async def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        """Generate answer using DeepSeek model via OpenRouter based on context chunks"""
        try:
            if not context_chunks:
                return "Information not available in the document."
            
            # Combine context chunks
            context = "\n\n".join(context_chunks)
            
            # Create prompt
            prompt = f"""
You are an AI assistant that answers questions based strictly on the provided document context. 

IMPORTANT INSTRUCTIONS:
1. Answer ONLY based on the information provided in the context below
2. If the information is not available in the context, respond with: "Information not available in the document."
3. Do not make assumptions or add information not present in the context
4. Provide direct, factual answers
5. Be concise but complete

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
            
            # Generate response using DeepSeek model via OpenRouter
            response = self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based strictly on provided document context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=settings.max_tokens,
                temperature=0.1,
                extra_headers={
                    "HTTP-Referer": "https://your-site.com",  # Optional: Replace with your site URL
                    "X-Title": "Bajaj AI QA System",  # Optional: Replace with your site name
                }
            )
            
            if response and response.choices and response.choices[0].message:
                answer = response.choices[0].message.content.strip()
                
                # Validate answer quality
                if self._is_valid_answer(answer):
                    return answer
                else:
                    return "Information not available in the document."
            else:
                return "Information not available in the document."
                
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "Information not available in the document."
    
    def _is_valid_answer(self, answer: str) -> bool:
        """Validate if the generated answer is appropriate"""
        # Check for common invalid responses
        invalid_phrases = [
            "i don't know",
            "i cannot",
            "not mentioned",
            "not specified",
            "unable to determine"
        ]
        
        answer_lower = answer.lower()
        
        # If answer contains invalid phrases, it's likely not found in context
        for phrase in invalid_phrases:
            if phrase in answer_lower:
                return False
        
        # Check minimum length
        if len(answer.strip()) < 10:
            return False
            
        return True