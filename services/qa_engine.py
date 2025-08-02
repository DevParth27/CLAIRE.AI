import openai
import logging
import asyncio
from typing import List, Dict
from config import settings
import re
from collections import Counter

logger = logging.getLogger(__name__)

class QAEngine:
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.max_retries = 3
        logger.info("Cost-Optimized High Accuracy QA Engine initialized")

    def estimate_tokens(self, text: str) -> int:
        """Accurate token estimation for cost control"""
        return int(len(text.split()) * 1.3)

    def organize_context(self, chunks: List[str], max_tokens: int = 8000) -> str:
        """Cost-optimized context organization for high accuracy"""
        if not chunks:
            return ""
        
        # Smart deduplication to reduce token usage
        unique_chunks = self._smart_deduplicate_chunks(chunks)
        prioritized_chunks = self._cost_effective_prioritize_chunks(unique_chunks)
        
        context_parts = []
        current_tokens = 0
        
        # Optimize for 8 high-quality chunks (cost-effective)
        for i, chunk in enumerate(prioritized_chunks[:8]):
            chunk_tokens = self.estimate_tokens(chunk)
            if current_tokens + chunk_tokens > max_tokens:
                break
            
            cleaned_chunk = self._efficient_clean_chunk(chunk)
            formatted_chunk = f"\n--- SECTION {i+1} ---\n{cleaned_chunk}"
            context_parts.append(formatted_chunk)
            current_tokens += chunk_tokens
        
        return "\n".join(context_parts)

    def _smart_deduplicate_chunks(self, chunks: List[str]) -> List[str]:
        """Efficient deduplication to reduce costs"""
        unique_chunks = []
        seen_hashes = set()
        
        for chunk in chunks:
            # Create content hash for quick comparison
            content_hash = hash(re.sub(r'\s+', ' ', chunk.lower().strip()))
            
            if content_hash not in seen_hashes:
                unique_chunks.append(chunk)
                seen_hashes.add(content_hash)
        
        return unique_chunks

    def _cost_effective_prioritize_chunks(self, chunks: List[str]) -> List[str]:
        """Cost-effective prioritization for maximum relevance"""
        scored_chunks = []
        
        for chunk in chunks:
            score = 0
            chunk_lower = chunk.lower()
            
            # High-impact keywords (focused list for efficiency)
            key_terms = [
                'coverage', 'premium', 'claim', 'benefit', 'waiting period',
                'maternity', 'pre-existing', 'exclusion', 'sum insured',
                'deductible', 'policy term', 'grace period'
            ]
            
            # Numerical patterns (high value for insurance)
            numerical_score = len(re.findall(r'\d+\s*%|\d+\s*years?|\d+\s*months?|rs\.?\s*\d+', chunk_lower))
            
            # Score calculation (optimized for relevance)
            for term in key_terms:
                score += chunk_lower.count(term) * 8
            
            score += numerical_score * 12
            score += min(len(chunk.split()) / 8, 15)  # Prefer substantial content
            
            scored_chunks.append((chunk, score))
        
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in scored_chunks]

    def _efficient_clean_chunk(self, chunk: str) -> str:
        """Efficient text cleaning for cost optimization"""
        # Essential cleaning only
        cleaned = re.sub(r'\s+', ' ', chunk)
        cleaned = re.sub(r'Rs\.?\s*', 'Rs. ', cleaned)
        cleaned = re.sub(r'(\d+)\s*%', r'\1%', cleaned)
        return cleaned.strip()

    # Update for better performance with cheaper models
    async def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        """Generate comprehensive answer with enhanced prompting and validation"""
        try:
            # Organize context efficiently
            context = self.organize_context(context_chunks, max_tokens=8000)
            
            if not context:
                return "I couldn't find relevant information in the document to answer your question."
            
            # Streamlined question analysis
            question_analysis = self._efficient_question_analysis(question)
            
            # Cost-optimized prompts
            system_prompt = self._create_cost_effective_system_prompt(question_analysis)
            user_prompt = self._create_efficient_user_prompt(question, context, question_analysis)
            
            # API call with cost optimization
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
                    formatted_answer = self._format_cost_effective_answer(answer, question_analysis)
                    
                    logger.info(f"Generated cost-effective answer for {question_analysis['type']} question")
                    return formatted_answer
                    
                except Exception as e:
                    logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                    if attempt == self.max_retries - 1:
                        raise e
                    await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I encountered an error while processing your question. Please try again."

    def _efficient_question_analysis(self, question: str) -> Dict[str, any]:
        """Streamlined question analysis for cost efficiency"""
        question_lower = question.lower()
        
        # Quick classification
        question_type = "factual"
        if any(word in question_lower for word in ['yes', 'no', 'does', 'is', 'can']):
            question_type = "yes_no"
        elif any(word in question_lower for word in ['how much', 'amount', 'cost']):
            question_type = "numerical"
        elif any(word in question_lower for word in ['when', 'period', 'duration']):
            question_type = "temporal"
        
        return {
            'type': question_type,
            'needs_numbers': any(word in question_lower for word in ['amount', 'cost', 'premium', '%']),
            'needs_time': any(word in question_lower for word in ['period', 'duration', 'months', 'years'])
        }

    def _create_cost_effective_system_prompt(self, question_analysis: Dict) -> str:
        """Concise system prompt for cost optimization"""
        prompt = """
You are an expert insurance document analyst. Provide accurate, concise answers based ONLY on the provided document context.

RULES:
1. Use ONLY information from the provided context
2. If information is missing, state "Not available in document"
3. Be precise with numbers, percentages, and time periods
4. Provide clear, direct answers
"""
        
        if question_analysis['type'] == 'yes_no':
            prompt += "\nFor YES/NO questions: Start with YES or NO, then explain briefly."
        elif question_analysis['type'] == 'numerical':
            prompt += "\nFor numerical questions: Provide exact figures with units/currency."
        elif question_analysis['type'] == 'temporal':
            prompt += "\nFor time questions: Provide exact periods as stated in document."
        
        return prompt

    def _create_efficient_user_prompt(self, question: str, context: str, question_analysis: Dict) -> str:
        """Efficient user prompt for cost optimization"""
        return f"""
DOCUMENT CONTEXT:
{context}

QUESTION: {question}

Provide a precise answer based on the document context above.
"""

    def _format_cost_effective_answer(self, answer: str, question_analysis: Dict) -> str:
        """Efficient answer formatting"""
        answer = answer.strip()
        
        # Ensure reasonable length (cost control)
        if len(answer) > 1200:
            sentences = answer.split('. ')
            truncated = []
            char_count = 0
            
            for sentence in sentences:
                if char_count + len(sentence) > 1100:
                    break
                truncated.append(sentence)
                char_count += len(sentence)
            
            answer = '. '.join(truncated)
            if not answer.endswith('.'):
                answer += '.'
        
        return answer