import google.generativeai as genai
import logging
import asyncio
import random
from typing import List, Dict, Optional
from config import settings
import re

logger = logging.getLogger(__name__)
qa_logger = logging.getLogger('qa_execution')

class EnhancedQAEngine:
    def __init__(self):
        # Configure Gemini
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(settings.gemini_model)
        self.model_name = settings.gemini_model
        self.max_retries = 3
        
        logger.info(f"Enhanced QA Engine initialized with {self.model_name}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using rough estimation"""
        # Fallback: rough estimation (1 token ≈ 4 characters)
        return len(text) // 4
    
    def organize_context(self, chunks: List[str], max_tokens: int = 12000) -> str:
        """Organize context chunks with better relevance scoring and token management"""
        if not chunks:
            return ""
        
        # Calculate relevance scores for each chunk
        scored_chunks = []
        for chunk in chunks:
            relevance_score = self._calculate_chunk_relevance(chunk)
            scored_chunks.append((chunk, relevance_score))
        
        # Sort by relevance score (descending)
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Build context within token limit
        context_parts = []
        current_tokens = 0
        
        for chunk, score in scored_chunks:
            chunk_tokens = self.count_tokens(chunk)
            if current_tokens + chunk_tokens <= max_tokens:
                context_parts.append(chunk)
                current_tokens += chunk_tokens
            else:
                # Try to fit partial chunk if there's significant space left
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 200:  # Only if substantial space remains
                    # Truncate chunk to fit
                    words = chunk.split()
                    partial_chunk = ""
                    for word in words:
                        test_chunk = partial_chunk + " " + word if partial_chunk else word
                        if self.count_tokens(test_chunk) <= remaining_tokens:
                            partial_chunk = test_chunk
                        else:
                            break
                    if partial_chunk:
                        context_parts.append(partial_chunk + "...")
                break
        
        # Join with clear separators
        context = "\n\n--- Document Section ---\n".join(context_parts)
        
        logger.info(f"Organized context: {len(context_parts)} chunks, {current_tokens} tokens")
        return context
    
    def _calculate_chunk_relevance(self, chunk: str) -> float:
        """Calculate relevance score for a chunk based on content quality indicators"""
        score = 0.0
        chunk_lower = chunk.lower()
        
        # Content quality indicators
        quality_indicators = [
            'policy', 'coverage', 'benefit', 'premium', 'claim', 'deductible',
            'exclusion', 'waiting period', 'sum insured', 'co-payment',
            'hospital', 'treatment', 'medical', 'diagnosis', 'procedure',
            'amount', 'limit', 'condition', 'requirement', 'eligible'
        ]
        
        # Score based on quality indicators
        for indicator in quality_indicators:
            if indicator in chunk_lower:
                score += 1.0
        
        # Bonus for structured content (lists, tables, definitions)
        if any(marker in chunk for marker in ['•', '-', '1.', '2.', ':', 'Rs.', '%']):
            score += 2.0
        
        # Bonus for specific numeric information
        if re.search(r'\d+', chunk):
            score += 1.0
        
        # Penalty for very short chunks (likely incomplete)
        if len(chunk) < 100:
            score -= 1.0
        
        # Bonus for medium-length chunks (likely complete thoughts)
        if 200 <= len(chunk) <= 800:
            score += 1.0
        
        # Penalty for very long chunks (might be less focused)
        if len(chunk) > 1500:
            score -= 0.5
        
        return max(0.0, score)  # Ensure non-negative score
    
    async def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        try:
            context = self.organize_context(context_chunks)
            prompt = self._build_enhanced_prompt(question, context)
            
            # Use Gemini API
            response = self.model.generate_content(prompt)
            answer = response.text
            
            return self._format_answer(answer, question)
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I apologize, but I encountered an error while generating an answer. Please try again."
    
    def _build_enhanced_prompt(self, question: str, context: str) -> str:
        """Build an enhanced prompt without few-shot examples"""
        
        # Determine if this is a math/calculation question
        is_math = self._is_math_question(question)
        
        if is_math:
            math_instruction = """
IMPORTANT: This question requires mathematical calculation. Please:
1. Extract the relevant numbers and formulas from the context
2. Show your calculation steps clearly
3. Provide the final numerical answer
4. Use the exact values mentioned in the document
"""
        else:
            math_instruction = ""
        
        prompt = f"""You are an expert document analyst. Answer the following question based ONLY on the provided context from the document.

CONTEXT FROM DOCUMENT:
{context}

QUESTION: {question}
{math_instruction}
INSTRUCTIONS:
- Answer based ONLY on the information provided in the context above
- Be specific and cite relevant details from the document
- If the information is not available in the context, clearly state that
- Provide a clear, well-structured answer
- For policy-related questions, include specific terms, conditions, and amounts when available
- If multiple scenarios exist, explain each one clearly

ANSWER:"""
        
        return prompt
    
    def _format_answer(self, answer: str, question: str) -> str:
        """Format and enhance the answer for better readability"""
        if not answer or len(answer.strip()) < 10:
            return "I couldn't find sufficient information in the document to answer your question."
        
        # Clean up the answer
        answer = answer.strip()
        
        # Remove any unwanted prefixes
        prefixes_to_remove = [
            "Based on the context provided:",
            "According to the document:",
            "From the information given:",
            "Answer:"
        ]
        
        for prefix in prefixes_to_remove:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
        
        # Ensure proper sentence structure
        if answer and not answer[0].isupper():
            answer = answer[0].upper() + answer[1:]
        
        # Ensure proper ending punctuation
        if answer and answer[-1] not in '.!?':
            answer += '.'
        
        return answer
    
    def _assess_answer_quality(self, answer: str, question: str, context: str) -> float:
        """Assess the quality of the generated answer"""
        if not answer or len(answer.strip()) < 10:
            return 0.0
        
        score = 0.0
        answer_lower = answer.lower()
        question_lower = question.lower()
        
        # Check if answer addresses the question
        question_keywords = set(re.findall(r'\b\w+\b', question_lower))
        answer_keywords = set(re.findall(r'\b\w+\b', answer_lower))
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 'how', 'when', 'where', 'why'}
        question_keywords -= stop_words
        answer_keywords -= stop_words
        
        # Keyword overlap score
        if question_keywords:
            overlap = len(question_keywords.intersection(answer_keywords))
            keyword_score = overlap / len(question_keywords)
            score += keyword_score * 30
        
        # Length appropriateness (not too short, not too long)
        answer_length = len(answer)
        if 50 <= answer_length <= 500:
            score += 20
        elif answer_length > 500:
            score += 10
        
        # Specificity indicators
        specificity_indicators = ['rs.', 'percent', '%', 'days', 'months', 'years', 'amount', 'limit']
        for indicator in specificity_indicators:
            if indicator in answer_lower:
                score += 5
        
        # Structure indicators
        if any(marker in answer for marker in [':', '-', '•', '1.', '2.']):
            score += 10
        
        # Avoid generic responses
        generic_phrases = ['i apologize', 'i cannot', 'not available', 'insufficient information']
        for phrase in generic_phrases:
            if phrase in answer_lower:
                score -= 15
        
        return min(100.0, max(0.0, score))
    
    def _is_math_question(self, question: str) -> bool:
        """Determine if a question requires mathematical calculation"""
        math_keywords = [
            'calculate', 'computation', 'total', 'sum', 'amount', 'cost',
            'premium', 'deductible', 'percentage', 'percent', '%',
            'multiply', 'divide', 'add', 'subtract', 'how much', 'how many'
        ]
        
        question_lower = question.lower()
        for keyword in math_keywords:
            if keyword in question_lower:
                return True
        
        return False