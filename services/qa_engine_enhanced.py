# import google.generativeai as genai
# import logging
# import asyncio
# import random
# from typing import List, Dict
# from config import settings

# # Import the training data manager
# from training_data import TrainingDataManager

# logger = logging.getLogger(__name__)
# qa_logger = logging.getLogger('qa_execution')

# class EnhancedQAEngine:
#     def __init__(self):
#         # Configure Gemini API
#         genai.configure(api_key=settings.gemini_api_key)
#         self.model = genai.GenerativeModel(
#             model_name=settings.gemini_model,
#             generation_config=genai.types.GenerationConfig(
#                 temperature=0.0,
#                 max_output_tokens=1000,
#                 top_p=0.95,
#                 top_k=40
#             )
#         )
#         self.max_retries = 3
        
#         # Load training data for few-shot examples
#         self.training_manager = TrainingDataManager()
#         logger.info("Enhanced QA Engine initialized with training examples")

#     def estimate_tokens(self, text: str) -> int:
#         """Estimate token count for text"""
#         return len(text.split()) * 1.3

#     def organize_context(self, chunks: List[str], max_tokens: int = 6000) -> str:
#         """Organize context chunks within token limit"""
#         context_parts = []
#         current_tokens = 0
        
#         # Use more chunks for better context
#         for chunk in chunks[:10]:
#             chunk_tokens = self.estimate_tokens(chunk)
#             if current_tokens + chunk_tokens > max_tokens:
#                 break
#             context_parts.append(chunk)
#             current_tokens += chunk_tokens
        
#         return "\n\n".join(context_parts)
    
#     def _get_few_shot_examples(self, question_type: str, count: int = 2) -> str:
#         """Get few-shot examples based on question type"""
#         examples = ""
        
#         # Determine question type
#         question_lower = question_type.lower()
#         category = None
        
#         if any(term in question_lower for term in ["premium", "claim", "policy", "coverage", "insurance", "medical", "health"]):
#             category = "policy_specific"
#         elif any(term in question_lower for term in ["article", "constitution", "law", "legal", "right"]):
#             category = "constitution_law"
#         elif any(term in question_lower for term in ["vehicle", "car", "brake", "tyre", "oil"]):
#             category = "vehicle_mechanical"
        
#         if category:
#             questions = self.training_manager.get_questions(category)
#             if questions and len(questions) >= count:
#                 selected = random.sample(questions, count)
#                 for q in selected:
#                     examples += f"Example Question: {q}\n"
#                     examples += "Example Answer: [Answer would be based on the policy document]\n\n"
        
#         return examples

#     async def generate_answer(self, question: str, context_chunks: List[str]) -> str:
#         """Generate comprehensive answer using Gemini API with few-shot examples"""
#         question_hash = str(hash(question))[-4:]
#         qa_logger.info(f"QA_START|Hash:{question_hash}|Question:{question[:50]}...")
        
#         try:
#             # Organize context
#             context = self.organize_context(context_chunks)
            
#             if not context:
#                 return "I couldn't find relevant information in the document to answer your question."
            
#             qa_logger.info(f"CONTEXT_ORGANIZED|Hash:{question_hash}|Tokens:{self.estimate_tokens(context)}")
            
#             # Get few-shot examples based on question type
#             few_shot_examples = self._get_few_shot_examples(question)
            
#             # Enhanced prompt with few-shot examples
#             prompt = f"""
#             You are an expert insurance policy analyst. Analyze the provided policy document context and answer questions with precision.

#             CRITICAL INSTRUCTIONS:
#             1. ONLY use information explicitly stated in the provided context
#             2. For specific details (waiting periods, coverage limits, percentages), quote exact values from the document
#             3. If information is not in the context, clearly state "This information is not mentioned in the provided document"
#             4. For yes/no questions, provide definitive answers based on the context
#             5. Include specific policy terms, conditions, and numerical values when available
#             6. Structure answers clearly with bullet points for multiple details
#             7. Be precise about coverage inclusions and exclusions
#             8. Always refer to the correct policy name mentioned in the document

#             {few_shot_examples}

#             Policy Document Context:
#             {context}

#             Question: {question}

#             Provide a precise answer based ONLY on the information in the context above. Include specific details, waiting periods, coverage limits, and conditions mentioned in the document.
#             """
            
#             qa_logger.info(f"API_CALL_START|Hash:{question_hash}|Attempt:1|Model:{settings.gemini_model}")
            
#             # Make API call with retries
#             for attempt in range(self.max_retries):
#                 try:
#                     start_time = asyncio.get_event_loop().time()
                    
#                     response = await asyncio.to_thread(
#                         self.model.generate_content,
#                         prompt
#                     )
                    
#                     api_time = asyncio.get_event_loop().time() - start_time
#                     answer = response.text.strip()
                    
#                     qa_logger.info(f"API_SUCCESS|Hash:{question_hash}|Attempt:{attempt+1}|Time:{api_time:.2f}s|Response_length:{len(answer)}")
                    
#                     formatted_answer = self._format_answer(answer)
#                     qa_logger.info(f"QA_COMPLETE|Hash:{question_hash}|Final_length:{len(formatted_answer)}")
                    
#                     return formatted_answer
                    
#                 except Exception as e:
#                     qa_logger.warning(f"Gemini API call attempt {attempt + 1} failed: {e}")
#                     if attempt == self.max_retries - 1:
#                         raise e
#                     await asyncio.sleep(1)
            
#         except Exception as e:
#             qa_logger.error(f"Error generating answer: {e}")
#             return "I encountered an error while processing your question. Please try again."
    
#     def _format_answer(self, answer: str) -> str:
#         """Format and clean up the generated answer"""
#         if not answer:
#             return "I couldn't generate an answer for your question."
        
#         answer = answer.strip()
        
#         # Allow longer answers for detailed policy information
#         if len(answer) > 1500:
#             answer = answer[:1497] + "..."
        
#         return answer

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
                temperature=0.05,  # Reduced for higher accuracy
                max_output_tokens=settings.max_tokens,  # Use config value
                top_p=0.95,  # Increased for better coverage
                top_k=40    # Increased for better coverage
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
        logger.info("Enhanced QA Engine initialized with training examples")

    def count_tokens(self, text: str) -> int:
        """Accurate token counting"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback estimation
            return int(len(text.split()) * 1.3)

    def organize_context(self, chunks: List[str], max_tokens: int = 8000) -> str:
        """Organize context chunks within token limit with better prioritization"""
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
        
        # Use configurable max context chunks
        for score, idx, chunk in scored_chunks[:settings.max_context_chunks]:
            chunk_tokens = self.count_tokens(chunk)
            if current_tokens + chunk_tokens > max_tokens:
                # Try to include partial chunk if it's highly relevant
                if score > 0.8 and current_tokens < max_tokens * 0.8:
                    remaining_tokens = max_tokens - current_tokens - 100  # Buffer
                    if remaining_tokens > 200:
                        partial_chunk = chunk[:int(remaining_tokens * 4)]  # Rough char estimate
                        context_parts.append(partial_chunk + "...")
                break
            context_parts.append(f"[Context {len(context_parts)+1}]: {chunk}")
            current_tokens += chunk_tokens
        
        context = "\n\n".join(context_parts)
        logger.info(f"Context organized: {len(context_parts)} chunks, ~{current_tokens} tokens")
        return context
    
    def _calculate_chunk_relevance(self, chunk: str) -> float:
        """Calculate relevance score for chunk prioritization"""
        score = 0.5  # Base score
        
        # Higher score for chunks with structured information
        if any(pattern in chunk.lower() for pattern in [
            'coverage', 'premium', 'deductible', 'limit', 'benefit',
            'exclusion', 'waiting period', 'policy', 'claim'
        ]):
            score += 0.3
        
        # Higher score for numerical data
        if re.search(r'\d+%|\$\d+|\d+ days|\d+ years', chunk):
            score += 0.2
        
        # Higher score for policy sections/headers
        if re.search(r'section \d+|article \d+|clause \d+', chunk.lower()):
            score += 0.1
        
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
        
        try:
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
        """Build enhanced prompt with better structure"""
        return f"""You are a highly skilled document analyst specializing in extracting precise information from policy documents, legal texts, and technical manuals.

ANALYSIS METHODOLOGY:
1. Read the entire context carefully before answering
2. Identify specific sections that directly address the question
3. Extract exact numerical values, percentages, time periods, and conditions
4. Cross-reference information across different context sections
5. Distinguish between general statements and specific policy provisions
6. Analyze the document structure to understand hierarchical relationships
7. Identify and interpret domain-specific terminology

RESPONSE REQUIREMENTS:
✓ ONLY use information explicitly stated in the provided context
✓ Quote exact values (percentages, amounts, time periods) with section references
✓ For yes/no questions, provide definitive answers with supporting evidence
✓ If information is incomplete, specify what details are missing
✓ Use bullet points for multiple related details
✓ Include relevant policy terms and conditions
✓ Cite specific document sections when available
✓ Maintain the exact terminology used in the document

ACCURACY STANDARDS:
- Exact numerical precision (don't round or approximate)
- Specific terminology from the document
- Clear distinction between covered and excluded items
- Accurate representation of conditions and requirements
- Precise citation of relevant sections

{few_shot_examples}

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

ANALYSIS AND ANSWER:
Based on my analysis of the provided document context, here is the precise answer:"""

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
        policy_terms = ['coverage', 'premium', 'deductible', 'policy', 'claim', 'benefit']
        if any(term in answer.lower() for term in policy_terms):
            score += 0.1
        
        # Penalize vague responses
        vague_phrases = ['may vary', 'depends on', 'generally', 'usually', 'typically']
        if any(phrase in answer.lower() for phrase in vague_phrases):
            score -= 0.1
        
        # Check for citation of document sections
        if any(phrase in answer.lower() for phrase in ['section', 'article', 'clause', 'according to']):
            score += 0.1
        
        return min(max(score, 0.0), 1.0)