import google.generativeai as genai
import logging
import asyncio
from typing import List, Dict
from config import settings

logger = logging.getLogger(__name__)
qa_logger = logging.getLogger('qa_execution')

class QAEngine:
    def __init__(self):
        # Configure Gemini API
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(
            model_name=settings.gemini_model,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=1000,
                top_p=0.95,
                top_k=40
            )
        )
        self.max_retries = 3
        logger.info("Cost-Optimized High Accuracy QA Engine initialized with Gemini")

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        return len(text.split()) * 1.3

    def organize_context(self, chunks: List[str], max_tokens: int = 6000) -> str:
        """Organize context chunks within token limit"""
        context_parts = []
        current_tokens = 0
        
        # Use more chunks for better context
        for chunk in chunks[:10]:
            chunk_tokens = self.estimate_tokens(chunk)
            if current_tokens + chunk_tokens > max_tokens:
                break
            context_parts.append(chunk)
            current_tokens += chunk_tokens
        
        return "\n\n".join(context_parts)

    async def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        """Generate comprehensive answer using Gemini API"""
        question_hash = str(hash(question))[-4:]
        qa_logger.info(f"QA_START|Hash:{question_hash}|Question:{question[:50]}...")
        
        try:
            # Organize context
            context = self.organize_context(context_chunks)
            
            if not context:
                return "I couldn't find relevant information in the document to answer your question."
            
            qa_logger.info(f"CONTEXT_ORGANIZED|Hash:{question_hash}|Tokens:{self.estimate_tokens(context)}")
            
            # Enhanced prompt for policy documents
            prompt = f"""
You are an expert insurance policy analyst. Analyze the provided policy document context and answer questions with precision.

CRITICAL INSTRUCTIONS:
1. ONLY use information explicitly stated in the provided context
2. For specific details (waiting periods, coverage limits, percentages), quote exact values from the document
3. If information is not in the context, clearly state "This information is not mentioned in the provided document"
4. For yes/no questions, provide definitive answers based on the context
5. Include specific policy terms, conditions, and numerical values when available
6. Structure answers clearly with bullet points for multiple details
7. Be precise about coverage inclusions and exclusions
8. Always refer to the correct policy name mentioned in the document

Policy Document Context:
{context}

Question: {question}

Provide a precise answer based ONLY on the information in the context above. Include specific details, waiting periods, coverage limits, and conditions mentioned in the document.
"""
            
            qa_logger.info(f"API_CALL_START|Hash:{question_hash}|Attempt:1|Model:{settings.gemini_model}")
            
            # Make API call with retries
            for attempt in range(self.max_retries):
                try:
                    start_time = asyncio.get_event_loop().time()
                    
                    response = await asyncio.to_thread(
                        self.model.generate_content,
                        prompt
                    )
                    
                    api_time = asyncio.get_event_loop().time() - start_time
                    answer = response.text.strip()
                    
                    qa_logger.info(f"API_SUCCESS|Hash:{question_hash}|Attempt:{attempt+1}|Time:{api_time:.2f}s|Response_length:{len(answer)}")
                    
                    formatted_answer = self._format_answer(answer)
                    qa_logger.info(f"QA_COMPLETE|Hash:{question_hash}|Final_length:{len(formatted_answer)}")
                    
                    return formatted_answer
                    
                except Exception as e:
                    qa_logger.warning(f"Gemini API call attempt {attempt + 1} failed: {e}")
                    if attempt == self.max_retries - 1:
                        raise e
                    await asyncio.sleep(1)
            
        except Exception as e:
            qa_logger.error(f"Error generating answer: {e}")
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