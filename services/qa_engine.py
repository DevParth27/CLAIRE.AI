import openai
import logging
import time
import asyncio
from typing import List
from config import settings

logger = logging.getLogger(__name__)

class QAEngine:
    def __init__(self):
        # Debug: Log the configuration (without exposing the full API key)
        logger.info(f"Initializing QA Engine with base_url: {settings.openai_base_url}")
        logger.info(f"API key starts with: {settings.openai_api_key[:10]}...")
        logger.info(f"Using model: {settings.openai_model}")
        
        # Configure OpenAI client
        try:
            self.client = openai.OpenAI(
                base_url=settings.openai_base_url,
                api_key=settings.openai_api_key
            )
        except TypeError as e:
            logger.warning(f"OpenAI client initialization failed with TypeError: {e}")
            logger.info("Attempting fallback initialization...")
            self.client = openai.OpenAI(
                api_key=settings.openai_api_key
            )
            if hasattr(self.client, '_base_url'):
                self.client._base_url = settings.openai_base_url

    def _estimate_tokens(self, text: str) -> int:
        """Accurate token estimation for GPT-4"""
        return int(len(text) // 3.2)  # More accurate for GPT-4

    def _organize_context_by_relevance(self, context_chunks: List[str], max_context_tokens: int = 6000) -> str:
        """Organize context to maximize information density and relevance"""
        if not context_chunks:
            return ""
        
        # Create organized sections with clear demarcation
        organized_sections = []
        current_tokens = 0
        
        for i, chunk in enumerate(context_chunks, 1):
            # Add section header for clarity
            section_header = f"\n=== POLICY SECTION {i} ===\n"
            section_content = f"{section_header}{chunk}"
            section_tokens = self._estimate_tokens(section_content)
            
            if current_tokens + section_tokens <= max_context_tokens:
                organized_sections.append(section_content)
                current_tokens += section_tokens
            else:
                # Try to fit a truncated version
                remaining_tokens = max_context_tokens - current_tokens
                if remaining_tokens > 400:  # Only if meaningful space
                    # Truncate at sentence boundaries
                    sentences = chunk.split('. ')
                    truncated_content = ""
                    
                    for sentence in sentences:
                        test_content = f"{section_header}{truncated_content}{sentence}. "
                        if self._estimate_tokens(test_content) <= remaining_tokens:
                            truncated_content += sentence + ". "
                        else:
                            break
                    
                    if truncated_content.strip():
                        organized_sections.append(f"{section_header}{truncated_content.strip()}...")
                break
        
        final_context = "".join(organized_sections)
        logger.info(f"Organized context: ~{self._estimate_tokens(final_context)} tokens from {len(organized_sections)} sections")
        return final_context

    async def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        """Generate ultra-concise, one-line answers"""
        max_retries = 2  # Reduced retries for speed
        
        for attempt in range(max_retries):
            try:
                if not context_chunks:
                    return "Information not found in document."
                
                # Use minimal context for speed
                context = "\n\n".join(context_chunks[:3])  # Only top 3 chunks
                
                # Ultra-concise prompt for one-line answers
                system_prompt = """Extract ONLY the specific factual answer in ONE sentence. Rules:
1. Maximum 20 words
2. Include exact numbers/timeframes when available
3. No explanations or elaborations
4. If not found, say "Not specified in document"
5. Be direct and factual only"""
                
                user_prompt = f"""Document: {context[:2000]}  # Limit context size

Question: {question}

One-line answer:"""
                
                # Generate response with speed optimizations
                response = self.client.chat.completions.create(
                    model=settings.openai_model,  # Now using gpt-3.5-turbo
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=50,  # Very small for one-line answers
                    temperature=0.0,
                    top_p=0.9,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                
                if response and response.choices and response.choices[0].message:
                    answer = response.choices[0].message.content.strip()
                    
                    # Ensure it's truly one line
                    answer = self._format_one_line_answer(answer)
                    
                    if len(answer.strip()) > 5:
                        return answer
                    else:
                        return "Information not available."
                else:
                    return "Unable to generate response."
                    
            except Exception as e:
                logger.error(f"Error generating answer: {str(e)}")
                if attempt < max_retries - 1:
                    continue
                else:
                    return "Error processing question."
        
        return "Unable to process question."
    
    def _format_one_line_answer(self, answer: str) -> str:
        """Ensure answer is exactly one line and concise"""
        # Remove extra whitespace and newlines
        answer = ' '.join(answer.split())
        
        # Take only the first sentence
        sentences = answer.split('. ')
        answer = sentences[0]
        
        # Ensure proper punctuation
        if not answer.endswith(('.', '!', '?')):
            answer += '.'
        
        # Limit to 25 words maximum
        words = answer.split()
        if len(words) > 25:
            answer = ' '.join(words[:25]) + '...'
        
        return answer

    def _organize_context_by_relevance(self, context_chunks: List[str], max_context_tokens: int = 6000) -> str:
        """Organize context to maximize information density and relevance"""
        if not context_chunks:
            return ""
        
        # Create organized sections with clear demarcation
        organized_sections = []
        current_tokens = 0
        
        for i, chunk in enumerate(context_chunks, 1):
            # Add section header for clarity
            section_header = f"\n=== POLICY SECTION {i} ===\n"
            section_content = f"{section_header}{chunk}"
            section_tokens = self._estimate_tokens(section_content)
            
            if current_tokens + section_tokens <= max_context_tokens:
                organized_sections.append(section_content)
                current_tokens += section_tokens
            else:
                # Try to fit a truncated version
                remaining_tokens = max_context_tokens - current_tokens
                if remaining_tokens > 400:  # Only if meaningful space
                    # Truncate at sentence boundaries
                    sentences = chunk.split('. ')
                    truncated_content = ""
                    
                    for sentence in sentences:
                        test_content = f"{section_header}{truncated_content}{sentence}. "
                        if self._estimate_tokens(test_content) <= remaining_tokens:
                            truncated_content += sentence + ". "
                        else:
                            break
                    
                    if truncated_content.strip():
                        organized_sections.append(f"{section_header}{truncated_content.strip()}...")
                break
        
        final_context = "".join(organized_sections)
        logger.info(f"Organized context: ~{self._estimate_tokens(final_context)} tokens from {len(organized_sections)} sections")
        return final_context

    async def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        """Generate concise, one-line answers"""
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                if not context_chunks:
                    return "No relevant information found in the document."
                
                # Use maximum context for comprehensive analysis
                max_context_tokens = 5500
                
                context = self._prepare_comprehensive_context(context_chunks, max_context_tokens)
                
                # Concise prompt for one-line answers
                system_prompt = """You are an expert insurance policy analyzer. Extract precise factual information and provide ONLY concise, one-line answers.

RULES:
1. Answer in ONE sentence maximum
2. Include specific numbers, timeframes, or amounts when available
3. Be direct and factual
4. No explanations or elaborations
5. If information is not found, state "Not specified in the document"
6. Focus on exact policy terms and conditions"""
                
                user_prompt = f"""INSURANCE POLICY DOCUMENT:
{context}

---

QUESTION: {question}

Provide a concise, one-line answer with specific details if available:"""
                
                # Estimate total tokens
                estimated_tokens = (
                    self._estimate_tokens(system_prompt) +
                    self._estimate_tokens(user_prompt) +
                    200 +  # Reduced max_tokens for concise answers
                    200  # overhead
                )
                
                logger.info(f"Estimated total tokens: {estimated_tokens}")
                
                # Generate response with settings optimized for concise answers
                response = self.client.chat.completions.create(
                    model=settings.openai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=200,  # Reduced for one-line answers
                    temperature=0.0,  # Completely deterministic
                    top_p=0.9,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                
                if response and response.choices and response.choices[0].message:
                    answer = response.choices[0].message.content.strip()
                    
                    # Format answer to ensure it's concise
                    answer = self._format_concise_answer(answer)
                    
                    # Log the answer for debugging
                    logger.info(f"Generated answer: {answer}")
                    
                    if len(answer.strip()) > 5:
                        return answer
                    else:
                        return "Unable to find specific information in the document."
                else:
                    return "Unable to generate response."
                    
            except openai.RateLimitError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Rate limit hit, retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"Rate limit exceeded after {max_retries} attempts")
                    return "Service temporarily unavailable. Please try again later."
                    
            except Exception as e:
                if "context_length_exceeded" in str(e):
                    max_context_tokens = max(3000, max_context_tokens - 1000)
                    logger.warning(f"Context too long, reducing to {max_context_tokens} tokens")
                    continue
                
                logger.error(f"Error generating answer: {str(e)}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                else:
                    return "Error processing question. Please try again."
        
        return "Unable to process question after multiple attempts."
    
    def _format_concise_answer(self, answer: str) -> str:
        """Format answer to ensure it's concise and one-line"""
        # Remove extra whitespace and newlines
        answer = ' '.join(answer.split())
        
        # Take only the first sentence if multiple sentences exist
        sentences = answer.split('. ')
        if len(sentences) > 1:
            answer = sentences[0] + '.'
        
        # Ensure it ends with proper punctuation
        if not answer.endswith(('.', '!', '?')):
            answer += '.'
        
        # Limit length to ensure it's truly concise (max 150 characters)
        if len(answer) > 150:
            answer = answer[:147] + '...'
        
        return answer

    def _organize_context_by_relevance(self, context_chunks: List[str], max_context_tokens: int = 6000) -> str:
        """Organize context to maximize information density and relevance"""
        if not context_chunks:
            return ""
        
        # Create organized sections with clear demarcation
        organized_sections = []
        current_tokens = 0
        
        for i, chunk in enumerate(context_chunks, 1):
            # Add section header for clarity
            section_header = f"\n=== POLICY SECTION {i} ===\n"
            section_content = f"{section_header}{chunk}"
            section_tokens = self._estimate_tokens(section_content)
            
            if current_tokens + section_tokens <= max_context_tokens:
                organized_sections.append(section_content)
                current_tokens += section_tokens
            else:
                # Try to fit a truncated version
                remaining_tokens = max_context_tokens - current_tokens
                if remaining_tokens > 400:  # Only if meaningful space
                    # Truncate at sentence boundaries
                    sentences = chunk.split('. ')
                    truncated_content = ""
                    
                    for sentence in sentences:
                        test_content = f"{section_header}{truncated_content}{sentence}. "
                        if self._estimate_tokens(test_content) <= remaining_tokens:
                            truncated_content += sentence + ". "
                        else:
                            break
                    
                    if truncated_content.strip():
                        organized_sections.append(f"{section_header}{truncated_content.strip()}...")
                break
        
        final_context = "".join(organized_sections)
        logger.info(f"Organized context: ~{self._estimate_tokens(final_context)} tokens from {len(organized_sections)} sections")
        return final_context

    async def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        """Generate production-quality comprehensive answers"""
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                if not context_chunks:
                    return "I couldn't find relevant information in the document to answer your question."
                
                # Allocate maximum context for comprehensive analysis
                max_context_tokens = 6000  # Generous allocation for detailed extraction
                
                context = self._organize_context_by_relevance(context_chunks, max_context_tokens)
                
                # Production-grade system prompt for insurance policy analysis
                system_prompt = """You are an expert insurance policy analyst specializing in extracting precise, comprehensive information from policy documents. Your responses must be detailed, accurate, and include specific policy terms.

EXTRACTION REQUIREMENTS:
1. Provide COMPLETE and SPECIFIC answers with exact timeframes, percentages, and conditions
2. Include relevant policy language and definitions when applicable
3. Extract ALL relevant details including sub-conditions, limits, and exceptions
4. Use exact numbers, percentages, and timeframes from the policy
5. Quote specific policy provisions when they directly answer the question
6. Provide comprehensive coverage of the topic, not just basic information
7. Include eligibility criteria, limitations, and special conditions

FORMAT REQUIREMENTS:
- Start with the direct answer
- Include specific details, numbers, and conditions
- Mention any relevant limitations or exceptions
- Use clear, professional language
- Ensure completeness and accuracy"""
                
                user_prompt = f"""INSURANCE POLICY DOCUMENT SECTIONS:
{context}

---

QUESTION: {question}

INSTRUCTIONS:
Analyze ALL policy sections above thoroughly. Extract comprehensive information related to this question including:
- Specific timeframes (days, months, years)
- Exact percentages, limits, and amounts
- Eligibility criteria and conditions
- Any exceptions or special provisions
- Relevant policy definitions
- Complete coverage details

Provide a detailed, comprehensive answer that includes all relevant information found in the policy sections.

COMPREHENSIVE ANSWER:"""
                
                # Calculate token usage
                estimated_tokens = (
                    self._estimate_tokens(system_prompt) +
                    self._estimate_tokens(user_prompt) +
                    settings.max_tokens +
                    300  # overhead
                )
                
                logger.info(f"Estimated total tokens: {estimated_tokens}")
                
                # Generate comprehensive response
                response = self.client.chat.completions.create(
                    model=settings.openai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=settings.max_tokens,
                    temperature=0.0,  # Deterministic for accuracy
                    top_p=0.95,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                
                if response and response.choices and response.choices[0].message:
                    answer = response.choices[0].message.content.strip()
                    
                    # Log answer quality metrics
                    logger.info(f"Generated answer length: {len(answer)} characters")
                    
                    # Minimal validation - accept comprehensive answers
                    if len(answer.strip()) > 20 and not self._is_generic_rejection(answer):
                        return answer
                    else:
                        logger.warning(f"Answer may be too generic: {answer[:100]}...")
                        return answer  # Return anyway for debugging
                else:
                    return "I couldn't generate a response. Please try again."
                    
            except openai.RateLimitError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Rate limit hit, retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"Rate limit exceeded after {max_retries} attempts")
                    return "Service temporarily unavailable due to rate limits. Please try again later."
                    
            except Exception as e:
                if "context_length_exceeded" in str(e):
                    # Reduce context more gradually
                    max_context_tokens = max(4000, max_context_tokens - 1000)
                    logger.warning(f"Context too long, reducing to {max_context_tokens} tokens")
                    continue
                
                logger.error(f"Error generating answer: {str(e)}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                else:
                    return "I encountered an error while processing your question. Please try again."
        
        return "I couldn't process your question after multiple attempts. Please try again."

    def _is_generic_rejection(self, answer: str) -> bool:
        """Check if answer is a generic rejection"""
        rejection_phrases = [
            "does not provide",
            "not available",
            "cannot find",
            "not specified",
            "information is not"
        ]
        
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in rejection_phrases)