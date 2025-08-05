import google.generativeai as genai
import logging
import asyncio
import random
from typing import List, Dict, Optional
from config import settings
import tiktoken  # Better token counting
import re

# Import the training data manager
from training_data import TrainingDataManager

logger = logging.getLogger(__name__)
qa_logger = logging.getLogger('qa_execution')

class EnhancedQAEngine:
    def __init__(self):
        # Configure Gemini API
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(
            model_name=settings.gemini_model,
            generation_config=genai.types.GenerationConfig(
                temperature=settings.temperature,  # Use from config
                max_output_tokens=4000,  # Increased for more comprehensive answers
                top_p=settings.top_p,    # Use from config
                top_k=settings.top_k     # Use from config
            )
        )
        self.max_retries = 3
        
        # Initialize tokenizer for better token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.tokenizer = None
            logger.warning("tiktoken not available, using fallback token estimation")
        
        # Load training data for few-shot examples
        self.training_manager = TrainingDataManager()
        logger.info(f"Enhanced QA Engine initialized with {settings.gemini_model}")

    def count_tokens(self, text: str) -> int:
        """Accurate token counting"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback estimation
            return int(len(text.split()) * 1.3)

    def organize_context(self, chunks: List[str], max_tokens: int = 12000) -> str:
        """Organize context chunks within token limit with better prioritization for any document type"""
        if not chunks:
            return ""
        
        # Score chunks by relevance indicators
        scored_chunks = []
        for i, chunk in enumerate(chunks):
            score = self._calculate_chunk_relevance(chunk)
            scored_chunks.append((score, i, chunk))
        
        # Sort by relevance score (descending)
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        context_parts = []
        current_tokens = 0
        
        # First pass: Include highest-scoring chunks
        for score, idx, chunk in scored_chunks[:10]:  # Include more top chunks
            chunk_tokens = self.count_tokens(chunk)
            if current_tokens + chunk_tokens > max_tokens * 0.6:  # Use more space for top chunks
                break
            context_parts.append(f"[SECTION {idx+1}]:\n{chunk}")
            current_tokens += chunk_tokens
        
        # Second pass: Include chunks with numerical data or key information
        for score, idx, chunk in scored_chunks:
            if f"[SECTION {idx+1}]" not in ''.join(context_parts):  # Skip already included
                # Enhanced regex for general information patterns
                if re.search(r'\d+%|\d+ days|\d+ months|\d+ years|\$\s*\d+|definition|defined as|refers to|means', chunk, re.IGNORECASE):
                    chunk_tokens = self.count_tokens(chunk)
                    if current_tokens + chunk_tokens > max_tokens * 0.85:  # Reserve less space
                        break
                    context_parts.append(f"[SECTION {idx+1}]:\n{chunk}")
                    current_tokens += chunk_tokens
                    if len(context_parts) >= 15:  # Increase chunk limit
                        break
        
        # Third pass: Fill remaining space with other chunks
        remaining_tokens = max_tokens - current_tokens - 200  # Buffer
        if remaining_tokens > 1000:
            for score, idx, chunk in scored_chunks:
                if f"[SECTION {idx+1}]" not in ''.join(context_parts):  # Skip already included
                    chunk_tokens = self.count_tokens(chunk)
                    if chunk_tokens > remaining_tokens:
                        continue
                    context_parts.append(f"[SECTION {idx+1}]:\n{chunk}")
                    remaining_tokens -= chunk_tokens
                    if remaining_tokens < 1000 or len(context_parts) >= 20:  # Increased hard limit
                        break
        
        context = "\n\n".join(context_parts)
        logger.info(f"Context organized: {len(context_parts)} chunks, ~{self.count_tokens(context)} tokens")
        return context
    
    def _calculate_chunk_relevance(self, chunk: str) -> float:
        """Calculate relevance score for chunk prioritization for any document type"""
        score = 0.5  # Base score
        
        # Higher score for chunks with definitional information
        if any(pattern in chunk.lower() for pattern in [
            'definition', 'defined as', 'refers to', 'means', 'is a', 'are the',
            'consists of', 'comprises', 'includes', 'excludes', 'limitations',
            'requirements', 'conditions', 'terms', 'specifications', 'guidelines'
        ]):
            score += 0.3
        
        # Higher score for numerical data
        if re.search(r'\d+%|\$\d+|\d+ days|\d+ years|\d+ months|\d+\.\d+', chunk):
            score += 0.2
        
        # Higher score for document sections/headers
        if re.search(r'section \d+|article \d+|clause \d+|chapter \d+|part \d+|paragraph \d+', chunk.lower()):
            score += 0.1
        
        # Higher score for question-answer patterns
        if re.search(r'\?|what is|how to|when|where|why|who|which|can|does|will', chunk.lower()):
            score += 0.15
        
        # Higher score for lists and enumerations
        if re.search(r'\d+\)|\d+\.|•|-\s|\*\s|\([a-z]\)|[ivxlcdm]+\)', chunk.lower()):
            score += 0.15
        
        return min(score, 1.0)
        """Calculate relevance score for chunk prioritization optimized for insurance policies"""
        score = 0.5  # Base score
        
        # Higher score for chunks with insurance-specific information
        if any(pattern in chunk.lower() for pattern in [
            'coverage', 'premium', 'deductible', 'limit', 'benefit',
            'exclusion', 'waiting period', 'policy', 'claim', 'grace period',
            'maternity', 'organ donor', 'ayush', 'room rent', 'icu', 'pre-existing'
        ]):
            score += 0.3
        
        # Higher score for numerical data relevant to insurance
        if re.search(r'\d+%|\$\d+|\d+ days|\d+ years|\d+ months|Rs\.?\s*\d+', chunk):
            score += 0.2
        
        # Higher score for policy sections/headers
        if re.search(r'section \d+|article \d+|clause \d+|table of benefits', chunk.lower()):
            score += 0.1
        
        # Higher score for specific question-related terms
        if any(term in chunk.lower() for term in [
            'grace period', 'pre-existing disease', 'ped', 'maternity', 
            'cataract', 'organ donor', 'no claim discount', 'ncd', 
            'health check', 'hospital', 'ayush', 'room rent', 'icu'
        ]):
            score += 0.4
        
        return min(score, 1.0)
    
    def _get_few_shot_examples(self, question: str, count: int = 3) -> str:
        """Get few-shot examples based on question analysis"""
        examples = ""
        
        # Enhanced question categorization
        question_lower = question.lower()
        category = self._categorize_question(question_lower)
        
        if category:
            questions = self.training_manager.get_questions(category)
            if questions and len(questions) >= count:
                selected = random.sample(questions, min(count, len(questions)))
                for i, q in enumerate(selected, 1):
                    examples += f"""
Example {i}:
Question: {q}
Answer: Based on the policy document, [specific answer with exact details from document sections, including numerical values, waiting periods, and conditions where applicable]

"""
        
        return examples

    def _categorize_question(self, question_lower: str) -> Optional[str]:
        """Enhanced question categorization"""
        categories = {
            "policy_specific": ["premium", "claim", "policy", "coverage", "insurance", 
                              "medical", "health", "benefit", "deductible", "copay"],
            "constitution_law": ["article", "constitution", "law", "legal", "right", 
                               "amendment", "judicial", "legislative"],
            "vehicle_mechanical": ["vehicle", "car", "brake", "tyre", "oil", "engine", 
                                 "maintenance", "mechanical", "automotive"],
            "financial": ["cost", "price", "fee", "payment", "billing", "finance", "money"],
            "procedural": ["how", "process", "procedure", "step", "apply", "submit", "file"]
        }
        
        for category, keywords in categories.items():
            if any(keyword in question_lower for keyword in keywords):
                return category
        
        return "general"

    async def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        """Generate comprehensive answer using Gemini API with enhanced prompting"""
        question_hash = str(hash(question))[-6:]  # Longer hash for better tracking
        qa_logger.info(f"QA_START|Hash:{question_hash}|Question:{question[:100]}...")
        
        # Check if this is a simple math question
        if self._is_math_question(question):
            try:
                # For simple math, we can bypass the context and directly query the model
                prompt = f"""You are a helpful AI assistant that can perform calculations accurately.

<question>
{question}
</question>

Provide a direct and accurate answer to this calculation question."""
                
                qa_logger.info(f"MATH_QUESTION_DETECTED|Hash:{question_hash}")
                
                response = await asyncio.to_thread(
                    self.model.generate_content,
                    prompt
                )
                
                answer = response.text.strip()
                return answer
                
            except Exception as e:
                qa_logger.error(f"Error processing math question: {e}")
                # Fall back to normal processing if math-specific handling fails
        
        try:
            # Regular document-based question processing
            # Organize context with better token management
            context = self.organize_context(context_chunks, max_tokens=10000)
            
            if not context:
                return "I couldn't find relevant information in the document to answer your question. Please ensure your question relates to the uploaded document content."
            
            context_tokens = self.count_tokens(context)
            qa_logger.info(f"CONTEXT_ORGANIZED|Hash:{question_hash}|Tokens:{context_tokens}")
            
            # Get few-shot examples
            few_shot_examples = self._get_few_shot_examples(question)
            
            # Enhanced prompt with better structure
            prompt = self._build_enhanced_prompt(question, context, few_shot_examples)
            
            qa_logger.info(f"API_CALL_START|Hash:{question_hash}|Model:{settings.gemini_model}")
            
            # Make API call with exponential backoff
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
                    
                    formatted_answer = self._format_answer(answer, question)
                    quality_score = self._assess_answer_quality(formatted_answer, question, context)
                    
                    qa_logger.info(f"QA_COMPLETE|Hash:{question_hash}|Quality:{quality_score:.2f}|Final_length:{len(formatted_answer)}")
                    
                    return formatted_answer
                    
                except Exception as e:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff
                    qa_logger.warning(f"Gemini API call attempt {attempt + 1} failed: {e}")
                    if attempt == self.max_retries - 1:
                        raise e
                    await asyncio.sleep(wait_time)
            
        except Exception as e:
            qa_logger.error(f"Error generating answer: {e}")
            return "I encountered an error while processing your question. Please try rephrasing your question or check if it relates to the uploaded document."
    
    def _build_enhanced_prompt(self, question: str, context: str, few_shot_examples: str) -> str:
        """Build enhanced prompt optimized for Gemini-2.5-flash for general-purpose QA"""
        
        # Detect if the question is a simple calculation
        if re.search(r'\d+\s*[+\-*/]\s*\d+', question.lower()):
            return f"""You are a helpful AI assistant that can perform calculations and answer questions accurately.

<question>
{question}
</question>

Provide a direct and accurate answer to this calculation question."""
        
        # For document-based questions
        return f"""You are a highly specialized document analysis AI with exceptional precision in extracting and interpreting information from any type of document.

<context>
{context}
</context>

<question>
{question}
</question>

<instructions>
1. ONLY use information explicitly stated in the provided context
2. Extract exact numerical values, percentages, time periods, and conditions when relevant
3. For specific details, quote exact values from the document
4. If information is not in the context, clearly state "This information is not mentioned in the provided document"
5. For yes/no questions, provide definitive answers based on the context
6. Include specific terms, conditions, and numerical values when available
7. Structure answers clearly with bullet points for multiple details
8. Be precise about inclusions and exclusions
9. Always refer to the correct terminology mentioned in the document
10. Maintain the exact terminology used in the document
11. NEVER make assumptions or inferences beyond what is explicitly stated
</instructions>

{few_shot_examples}

<answer_format>
- Start with a direct answer to the question
- Include specific section references when available
- Use bullet points for multiple related details
- Quote exact values and terminology from the document
- Maintain factual accuracy without assumptions
</answer_format>

Provide your precise answer based ONLY on the information in the context:"""

    def _format_answer(self, answer: str, question: str) -> str:
        """Enhanced answer formatting with quality checks"""
        if not answer:
            return "I couldn't generate an answer for your question based on the provided document."
        
        answer = answer.strip()
        
        # Remove redundant phrases
        redundant_phrases = [
            "Based on my analysis of the provided document context, here is the precise answer:",
            "ANALYSIS AND ANSWER:",
            "Based on the document context provided:",
        ]
        
        for phrase in redundant_phrases:
            answer = answer.replace(phrase, "").strip()
        
        # Ensure answer starts properly
        if not answer.startswith(("Yes", "No", "The", "According", "Based", "This", "Coverage", "Premium", "Policy")):
            if "not mentioned" in answer.lower() or "not provided" in answer.lower():
                answer = "This information is not mentioned in the provided document. " + answer
        
        # Smart truncation at sentence boundaries
        if len(answer) > settings.max_tokens * 4:  # Rough char to token ratio
            sentences = answer.split('. ')
            truncated = []
            char_count = 0
            
            for sentence in sentences:
                if char_count + len(sentence) > settings.max_tokens * 3.5:
                    break
                truncated.append(sentence)
                char_count += len(sentence)
            
            answer = '. '.join(truncated)
            if not answer.endswith('.'):
                answer += '.'
        
        return answer
    
    def _assess_answer_quality(self, answer: str, question: str, context: str) -> float:
        """Assess answer quality for monitoring"""
        score = 0.5  # Base score
        
        # Check for specific details
        if re.search(r'\d+%|\$\d+|\d+ days|\d+ years|\d+ months', answer):
            score += 0.2
        
        # Check for structured response
        if any(marker in answer for marker in ['•', '-', '1.', '2.', 'Yes,', 'No,']):
            score += 0.1
        
        # Check for policy-specific terms
        policy_terms = ['coverage', 'premium', 'deductible', 'policy', 'claim', 'benefit', 'waiting period', 'grace period']
        if any(term in answer.lower() for term in policy_terms):
            score += 0.1
        
        # Penalize vague responses
        vague_phrases = ['may vary', 'depends on', 'generally', 'usually', 'typically', 'might be', 'could be']
        if any(phrase in answer.lower() for phrase in vague_phrases):
            score -= 0.2  # Increased penalty
        
        # Check for citation of document sections
        if any(phrase in answer.lower() for phrase in ['section', 'article', 'clause', 'according to', 'as stated in']):
            score += 0.1
        
        # Penalize "not mentioned" responses when context likely contains the answer
        if "not mentioned" in answer.lower() or "not provided" in answer.lower():
            # Check if key terms from question appear in context
            question_terms = set(re.findall(r'\b\w{4,}\b', question.lower()))
            context_lower = context.lower()
            matching_terms = sum(1 for term in question_terms if term in context_lower)
            if matching_terms >= 3:  # If several question terms are in context
                score -= 0.3  # Heavy penalty for likely incorrect "not mentioned" response
        
        return min(max(score, 0.0), 1.0)

    def _is_math_question(self, question: str) -> bool:
        """Detect if the question is a simple math calculation"""
        # Check for basic arithmetic patterns
        if re.search(r'\d+\s*[+\-*/]\s*\d+', question.lower()):
            return True
            
        # Check for common math question phrases
        math_phrases = [
            'calculate', 'compute', 'what is the sum of', 'what is the product of',
            'what is the difference between', 'what is the result of', 'what is the value of',
            'solve for', 'evaluate'
        ]
        
        if any(phrase in question.lower() for phrase in math_phrases) and re.search(r'\d+', question):
            return True
            
        return False