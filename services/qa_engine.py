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
        logger.info("Enhanced QA Engine initialized successfully")

    def estimate_tokens(self, text: str) -> int:
        """More accurate token estimation for GPT models"""
        # More accurate estimation: ~1.3 tokens per word for English
        return int(len(text.split()) * 1.3)

    def organize_context(self, chunks: List[str], max_tokens: int = 6000) -> str:
        """Intelligently organize context chunks for maximum relevance"""
        if not chunks:
            return ""
        
        # Step 1: Deduplicate similar chunks
        unique_chunks = self._deduplicate_chunks(chunks)
        
        # Step 2: Prioritize chunks based on content quality
        prioritized_chunks = self._prioritize_chunks(unique_chunks)
        
        # Step 3: Organize chunks with clear section markers
        context_parts = []
        current_tokens = 0
        
        for i, chunk in enumerate(prioritized_chunks[:12]):  # Max 12 chunks
            chunk_tokens = self.estimate_tokens(chunk)
            if current_tokens + chunk_tokens > max_tokens:
                break
            
            # Clean and format chunk
            cleaned_chunk = self._clean_chunk(chunk)
            
            # Add section markers for clarity
            formatted_chunk = f"\n--- CONTEXT SECTION {i+1} ---\n{cleaned_chunk}"
            context_parts.append(formatted_chunk)
            current_tokens += chunk_tokens
        
        organized_context = "\n".join(context_parts)
        logger.info(f"Organized {len(context_parts)} context sections ({current_tokens} estimated tokens)")
        return organized_context
    
    def _deduplicate_chunks(self, chunks: List[str]) -> List[str]:
        """Remove highly similar chunks to avoid redundancy"""
        unique_chunks = []
        
        for chunk in chunks:
            is_duplicate = False
            chunk_words = set(chunk.lower().split())
            
            for existing_chunk in unique_chunks:
                existing_words = set(existing_chunk.lower().split())
                
                # Calculate Jaccard similarity
                intersection = len(chunk_words.intersection(existing_words))
                union = len(chunk_words.union(existing_words))
                similarity = intersection / union if union > 0 else 0
                
                if similarity > 0.75:  # 75% similarity threshold
                    is_duplicate = True
                    # Keep the longer chunk if it's a duplicate
                    if len(chunk) > len(existing_chunk):
                        unique_chunks[unique_chunks.index(existing_chunk)] = chunk
                    break
            
            if not is_duplicate:
                unique_chunks.append(chunk)
        
        logger.info(f"Deduplicated {len(chunks)} chunks to {len(unique_chunks)} unique chunks")
        return unique_chunks
    
    def _prioritize_chunks(self, chunks: List[str]) -> List[str]:
        """Prioritize chunks based on relevance indicators and content quality"""
        # Define priority keywords with weights
        priority_keywords = {
            # High priority - specific policy terms
            'waiting period': 3.0, 'grace period': 3.0, 'pre-existing disease': 3.0,
            'maternity': 2.5, 'cataract': 2.5, 'organ donor': 2.5,
            'no claim discount': 2.5, 'ayush': 2.5,
            
            # Medium priority - general policy terms
            'coverage': 2.0, 'benefit': 2.0, 'premium': 2.0, 'claim': 2.0,
            'hospital': 1.8, 'treatment': 1.8, 'room rent': 2.0, 'icu': 2.0,
            
            # Lower priority - common terms
            'policy': 1.5, 'insurance': 1.5, 'health': 1.2
        }
        
        scored_chunks = []
        
        for chunk in chunks:
            chunk_lower = chunk.lower()
            score = 0
            
            # Score based on keyword presence
            for keyword, weight in priority_keywords.items():
                if keyword in chunk_lower:
                    score += weight
            
            # Boost score for chunks with specific numerical information
            numerical_patterns = [
                r'\d+\s*(?:days?|months?|years?)',  # Time periods
                r'\d+\s*%',  # Percentages
                r'\d+\s*(?:rupees?|rs\.?|inr)',  # Amounts
                r'\d+\s*(?:years?|months?)\s*(?:waiting|grace)',  # Specific waiting periods
            ]
            
            for pattern in numerical_patterns:
                matches = len(re.findall(pattern, chunk_lower))
                score += matches * 1.5
            
            # Boost score for chunks with policy structure indicators
            structure_indicators = ['section', 'clause', 'article', 'part', 'chapter']
            for indicator in structure_indicators:
                if indicator in chunk_lower:
                    score += 0.5
            
            # Boost score for chunks with definitive language
            definitive_terms = ['shall', 'will', 'must', 'required', 'entitled', 'covered']
            definitive_count = sum(1 for term in definitive_terms if term in chunk_lower)
            score += definitive_count * 0.3
            
            # Penalize very short chunks
            if len(chunk.split()) < 20:
                score *= 0.5
            
            # Penalize very long chunks (might be poorly segmented)
            if len(chunk.split()) > 500:
                score *= 0.8
            
            scored_chunks.append((chunk, score))
        
        # Sort by score (descending) and return chunks
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        prioritized = [chunk for chunk, score in scored_chunks]
        
        logger.info(f"Prioritized chunks with scores: {[f'{score:.2f}' for _, score in scored_chunks[:5]]}")
        return prioritized
    
    def _clean_chunk(self, chunk: str) -> str:
        """Clean and format chunk for better readability"""
        # Remove excessive whitespace
        chunk = re.sub(r'\s+', ' ', chunk)
        
        # Remove context markers if present
        chunk = re.sub(r'\[(?:MAIN|CONTEXT)\]\s*', '', chunk)
        
        # Ensure proper sentence endings
        chunk = chunk.strip()
        if chunk and not chunk.endswith(('.', '!', '?', ':')):
            chunk += '.'
        
        return chunk

    # Update for better performance with cheaper models
    async def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        try:
            # Organize context intelligently
            context = self.organize_context(context_chunks)
            
            if not context:
                return "I couldn't find relevant information in the document to answer your question."
            
            # Analyze question for better prompting
            question_analysis = self._analyze_question(question)
            
            # Create enhanced system prompt
            system_prompt = self._create_system_prompt(question_analysis)
            
            # Create targeted user prompt
            user_prompt = self._create_user_prompt(question, context, question_analysis)
            
            # Make API call with retries
            for attempt in range(self.max_retries):
                try:
                    response = await asyncio.to_thread(
                        self.client.chat.completions.create,
                        model=settings.openai_model,  # Will use gpt-4-turbo
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=settings.max_tokens,
                        temperature=0.1,  # Slightly higher for creativity
                        top_p=0.9,       # More diverse responses
                        frequency_penalty=0.1,  # Reduce repetition
                        presence_penalty=0.1,   # Encourage new topics
                        timeout=30
                    )
                    
                    answer = response.choices[0].message.content.strip()
                    
                    # Validate and format answer
                    formatted_answer = self._format_and_validate_answer(answer, question_analysis)
                    
                    logger.info(f"Generated answer for {question_analysis['type']} question")
                    return formatted_answer
                    
                except Exception as e:
                    logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                    if attempt == self.max_retries - 1:
                        raise e
                    await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I encountered an error while processing your question. Please try again."
    
    def _analyze_question(self, question: str) -> Dict[str, any]:
        """Comprehensive question analysis for better answer generation"""
        question_lower = question.lower()
        
        analysis = {
            'original': question,
            'type': 'general',
            'expects_number': False,
            'expects_yes_no': False,
            'expects_list': False,
            'key_terms': [],
            'temporal_terms': [],
            'numerical_terms': []
        }
        
        # Determine question type
        if any(word in question_lower for word in ['what is', 'how much', 'how long', 'how many']):
            analysis['type'] = 'factual'
            analysis['expects_number'] = bool(re.search(r'how (?:much|long|many)', question_lower))
        elif any(word in question_lower for word in ['does', 'is there', 'are there', 'can', 'will']):
            analysis['type'] = 'yes_no'
            analysis['expects_yes_no'] = True
        elif 'waiting period' in question_lower or 'grace period' in question_lower:
            analysis['type'] = 'waiting_period'
            analysis['expects_number'] = True
        elif any(word in question_lower for word in ['cover', 'coverage', 'benefit', 'include']):
            analysis['type'] = 'coverage'
        elif any(word in question_lower for word in ['how', 'define', 'definition', 'what does']):
            analysis['type'] = 'definition'
        elif any(word in question_lower for word in ['list', 'what are', 'which']):
            analysis['type'] = 'list'
            analysis['expects_list'] = True
        
        # Extract key terms
        insurance_terms = [
            'waiting period', 'grace period', 'pre-existing disease', 'maternity',
            'cataract', 'organ donor', 'no claim discount', 'ayush', 'room rent',
            'icu charges', 'health check-up', 'premium', 'coverage', 'benefit'
        ]
        
        for term in insurance_terms:
            if term in question_lower:
                analysis['key_terms'].append(term)
        
        # Extract temporal terms
        temporal_matches = re.findall(r'\d+\s*(?:days?|months?|years?)', question_lower)
        analysis['temporal_terms'] = temporal_matches
        
        # Extract numerical terms
        numerical_matches = re.findall(r'\d+(?:\.\d+)?\s*%?', question_lower)
        analysis['numerical_terms'] = numerical_matches
        
        return analysis
    
    def _create_system_prompt(self, question_analysis: Dict) -> str:
        """Create targeted system prompt based on question analysis"""
        base_prompt = """
You are an expert insurance policy analyst with deep knowledge of health insurance policies and Indian insurance regulations.

CRITICAL INSTRUCTIONS:
1. ONLY use information explicitly stated in the provided context sections
2. For specific details (waiting periods, coverage limits, percentages, amounts), quote EXACT values from the document
3. If information is not in the context, clearly state "This information is not mentioned in the provided document sections"
4. Be precise about coverage inclusions and exclusions
5. When mentioning time periods, always include the exact duration
6. Structure answers clearly and logically
"""
        
        # Add question-type specific instructions
        if question_analysis['type'] == 'yes_no':
            base_prompt += """
7. For yes/no questions, start with a clear YES or NO, then provide supporting details
8. If the answer is conditional, explain the conditions clearly
"""
        elif question_analysis['type'] == 'waiting_period':
            base_prompt += """
7. For waiting period questions, provide the exact time duration (days/months/years)
8. Distinguish between different types of waiting periods if applicable
9. Include any conditions or exceptions to the waiting period
"""
        elif question_analysis['type'] == 'coverage':
            base_prompt += """
7. For coverage questions, clearly state what IS covered and what IS NOT covered
8. Include any sub-limits, conditions, or restrictions
9. Mention any waiting periods that apply to the coverage
"""
        elif question_analysis['type'] == 'factual' and question_analysis['expects_number']:
            base_prompt += """
7. For numerical questions, provide exact figures from the document
8. Include the unit of measurement (days, months, years, percentage, rupees)
9. If there are different values for different scenarios, list them clearly
"""
        elif question_analysis['expects_list']:
            base_prompt += """
7. For list questions, provide information in a clear, numbered or bulleted format
8. Include all relevant items mentioned in the context
9. If the list is incomplete in the context, mention this limitation
"""
        
        return base_prompt
    
    def _create_user_prompt(self, question: str, context: str, question_analysis: Dict) -> str:
        """Create targeted user prompt with context and question"""
        prompt = f"""
Policy Document Context:
{context}

Question Type: {question_analysis['type'].title()}
Key Terms Identified: {', '.join(question_analysis['key_terms']) if question_analysis['key_terms'] else 'None'}

Question: {question}

Provide a precise, comprehensive answer based ONLY on the information in the context sections above.
"""
        
        # Add specific instructions based on question type
        if question_analysis['expects_yes_no']:
            prompt += "\nStart your answer with YES or NO, then provide detailed explanation."
        elif question_analysis['expects_number']:
            prompt += "\nInclude specific numerical values with units (days, months, years, %, etc.)."
        elif question_analysis['expects_list']:
            prompt += "\nFormat your answer as a clear list with bullet points or numbers."
        
        prompt += "\n\nAnswer:"
        
        return prompt
    
    def _format_and_validate_answer(self, answer: str, question_analysis: Dict) -> str:
        """Format and validate the generated answer"""
        if not answer:
            return "I couldn't generate an answer for your question."
        
        answer = answer.strip()
        
        # Validate answer based on question type
        if question_analysis['expects_yes_no']:
            if not any(answer.upper().startswith(word) for word in ['YES', 'NO']):
                # Try to infer yes/no from content
                if any(word in answer.lower() for word in ['is covered', 'does cover', 'includes', 'entitled']):
                    answer = "YES. " + answer
                elif any(word in answer.lower() for word in ['not covered', 'does not cover', 'excludes', 'not entitled']):
                    answer = "NO. " + answer
        
        # Ensure proper sentence endings
        if answer and not answer.endswith(('.', '!', '?')):
            answer += "."
        
        # Limit answer length but allow for detailed policy information
        if len(answer) > 2500:
            answer = answer[:2497] + "..."
        
        return answer