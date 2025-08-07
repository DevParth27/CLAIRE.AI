import google.generativeai as genai
import logging
import asyncio
import re
import tiktoken
from typing import List, Dict, Any, Optional, Tuple
from config import settings
from services.vector_store import VectorStore

logger = logging.getLogger(__name__)
qa_logger = logging.getLogger('qa_execution')

class QAEngine:
    def __init__(self):
        # Configure Gemini API
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(
            model_name=settings.gemini_model,
            generation_config=genai.types.GenerationConfig(
                temperature=settings.temperature,
                max_output_tokens=800,  # Reduced from 1000
                top_p=settings.top_p,
                top_k=settings.top_k
            )
        )
        self.max_retries = 3
        
        # Initialize tokenizer for accurate token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize vector store
        self.vector_store = None
        
        logger.info(f"Enhanced QA Engine initialized with {settings.gemini_model}")

    def count_tokens(self, text: str) -> int:
        """Count tokens accurately using tiktoken"""
        return len(self.tokenizer.encode(text))

    def organize_context(self, chunks: List[str], max_tokens: int = 4000) -> str:
        """Organize context chunks within token limit with prioritization"""
        # Reduced from 6000 to 4000 max tokens
        if not chunks:
            return ""
            
        # Prioritize chunks with numerical data and key information
        prioritized_chunks = []
        regular_chunks = []
        
        for chunk in chunks:
            # Check for numerical data or key policy terms
            if re.search(r'\d+\s*(days?|months?|years?|%|percent)', chunk, re.IGNORECASE) or \
               any(term in chunk.lower() for term in ['waiting period', 'grace period', 'coverage', 'exclusion']):
                prioritized_chunks.append(chunk)
            else:
                regular_chunks.append(chunk)
        
        # Combine prioritized chunks first, then regular chunks
        ordered_chunks = prioritized_chunks + regular_chunks
        
        # Track token count
        context_parts = []
        current_tokens = 0
        
        for chunk in ordered_chunks:
            chunk_tokens = self.count_tokens(chunk)
            if current_tokens + chunk_tokens > max_tokens:
                break
            context_parts.append(chunk)
            current_tokens += chunk_tokens
        
        logger.info(f"Organized {len(context_parts)} chunks within {current_tokens} tokens")
        return "\n\n".join(context_parts)

    def categorize_question(self, question: str) -> str:
        """Categorize question to optimize retrieval and answer generation"""
        question_lower = question.lower()
        
        # Define categories and their patterns
        categories = {
            "policy_specific": ["policy", "coverage", "exclusion", "benefit", "premium"],
            "time_related": ["period", "waiting", "grace", "days", "months", "years"],
            "financial": ["amount", "limit", "sum", "cost", "price", "rupees", "rs", "inr", "%", "percent"],
            "procedural": ["how to", "process", "claim", "procedure", "submit", "apply"],
            "general": []
        }
        
        # Check each category
        for category, patterns in categories.items():
            if any(pattern in question_lower for pattern in patterns):
                return category
        
        return "general"

    def is_math_question(self, question: str) -> bool:
        """Detect if the question is a simple math question"""
        # Check for arithmetic patterns
        if re.search(r'\d+\s*[+\-*/]\s*\d+', question):
            return True
            
        # Check for common math phrases
        math_phrases = ["calculate", "compute", "sum of", "product of", "divide", "multiply"]
        if any(phrase in question.lower() for phrase in math_phrases):
            return True
            
        return False

    async def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        """Generate comprehensive answer using Gemini API with enhanced context handling"""
        question_hash = str(hash(question))[-4:]
        qa_logger.info(f"QA_START|Hash:{question_hash}|Question:{question[:50]}...")
        
        try:
            # Check if it's a simple math question (bypass context for direct model query)
            if self.is_math_question(question):
                qa_logger.info(f"MATH_QUESTION_DETECTED|Hash:{question_hash}")
                prompt = f"Calculate the following: {question}"
                response = await asyncio.to_thread(self.model.generate_content, prompt)
                return self._format_answer(response.text.strip())
            
            # Organize context with token management
            context = self.organize_context(context_chunks, max_tokens=settings.max_tokens)
            
            if not context:
                return "I couldn't find relevant information in the document to answer your question."
            
            qa_logger.info(f"CONTEXT_ORGANIZED|Hash:{question_hash}|Tokens:{self.count_tokens(context)}")
            
            # Categorize question for optimized prompt building
            question_category = self.categorize_question(question)
            qa_logger.info(f"QUESTION_CATEGORY|Hash:{question_hash}|Category:{question_category}")
            
            # Build enhanced prompt
            prompt = self._build_enhanced_prompt(question, context, question_category)
            
            qa_logger.info(f"API_CALL_START|Hash:{question_hash}|Attempt:1|Model:{settings.gemini_model}")
            
            # Make API call with retries and exponential backoff
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
                    
                    # Format and assess answer quality
                    formatted_answer = self._format_answer(answer)
                    answer_quality = self._assess_answer_quality(formatted_answer, question, context)
                    
                    qa_logger.info(f"QA_COMPLETE|Hash:{question_hash}|Final_length:{len(formatted_answer)}|Quality:{answer_quality:.2f}")
                    
                    return formatted_answer
                    
                except Exception as e:
                    qa_logger.warning(f"Gemini API call attempt {attempt + 1} failed: {e}")
                    if attempt == self.max_retries - 1:
                        raise e
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)
            
        except Exception as e:
            qa_logger.error(f"Error generating answer: {e}")
            return "I encountered an error while processing your question. Please try again."
    
    def _build_enhanced_prompt(self, question: str, context: str, category: str) -> str:
        """Build enhanced prompt based on question category"""
        # For simple calculations, use direct prompt
        if self.is_math_question(question):
            return f"Calculate the following: {question}"
        
        # Build prompt based on document context
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
        
        # Add category-specific instructions
        if category == "time_related":
            prompt += "\nBe extremely precise about time periods, waiting periods, and grace periods mentioned in the document. Quote the exact number of days/months/years."
        elif category == "financial":
            prompt += "\nFocus on financial details, coverage limits, percentages, and monetary values. Quote exact figures from the document."
        elif category == "procedural":
            prompt += "\nClearly outline the steps or procedures mentioned in the document. Present them in a sequential, easy-to-follow format."
        
        return prompt
    
    def _format_answer(self, answer: str) -> str:
        """Format and clean up the generated answer"""
        if not answer:
            return "I couldn't generate an answer for your question."
        
        answer = answer.strip()
        
        # Allow longer answers for detailed policy information
        if len(answer) > 1500:
            answer = answer[:1497] + "..."
        
        return answer
    
    def _assess_answer_quality(self, answer: str, question: str, context: str) -> float:
        """Assess the quality of the generated answer"""
        score = 0.5  # Base score
        
        # Check for specific details
        if re.search(r'\d+\s*(days?|months?|years?|%|percent)', answer, re.IGNORECASE):
            score += 0.1
        
        # Check for structured response
        if re.search(r'\n\s*[-•*]\s', answer):
            score += 0.1
        
        # Check for policy-specific terms
        policy_terms = ['coverage', 'benefit', 'waiting', 'grace', 'claim', 'premium', 'policy']
        term_count = sum(1 for term in policy_terms if term in answer.lower())
        score += min(0.2, term_count * 0.03)
        
        # Penalize vague phrases
        vague_phrases = ['might be', 'could be', 'possibly', 'perhaps', 'may have']
        vague_count = sum(1 for phrase in vague_phrases if phrase in answer.lower())
        score -= min(0.2, vague_count * 0.05)
        
        # Penalize "not mentioned" responses if key terms from question are in context
        if "not mentioned" in answer.lower() or "not provided" in answer.lower():
            question_keywords = set(re.findall(r'\b\w+\b', question.lower()))
            context_keywords = set(re.findall(r'\b\w+\b', context.lower()))
            overlap = len(question_keywords.intersection(context_keywords))
            
            if overlap > 3:  # If significant overlap exists
                score -= 0.2
        
        return max(0.1, min(1.0, score))  # Ensure score is between 0.1 and 1.0

    async def set_vector_store(self, vector_store: VectorStore):
        """Set the vector store for direct access"""
        self.vector_store = vector_store
        logger.info("Vector store connected to QA Engine")
        
    async def search_and_answer(self, question: str, document_id: str) -> str:
        """Integrated search and answer generation"""
        if not self.vector_store:
            return "Vector store not connected. Please initialize properly."
            
        try:
            # Search for relevant chunks
            chunks = await self.vector_store.search_similar(question, document_id)
            
            if not chunks:
                return "I couldn't find relevant information in the document to answer your question."
                
            # Generate answer
            return await self.generate_answer(question, chunks)
            
        except Exception as e:
            logger.error(f"Error in search_and_answer: {str(e)}")
            return f"Error processing your question: {str(e)}"