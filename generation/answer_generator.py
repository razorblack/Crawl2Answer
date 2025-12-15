"""
Answer Generation Module for Crawl2Answer RAG System

This module provides comprehensive answer generation capabilities
using retrieved context and language models. It prepares prompts,
calls language models, and ensures answers are grounded in the
provided context.

Features:
- Context-aware prompt generation
- Multiple language model support
- Source tracking and citation
- Answer validation and filtering
- Response formatting and post-processing
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
import os
from abc import ABC, abstractmethod

from retrieval.retriever_enhanced import DocumentRetriever, RetrievalResult
from config.settings import Settings

logger = logging.getLogger(__name__)

@dataclass
class AnswerResult:
    """Result from answer generation"""
    question: str
    answer: str
    sources: List[str]
    context_used: str
    confidence_score: float
    generation_time: float
    model_used: str
    chunk_count: int
    retrieval_scores: List[float]
    
    def get_formatted_answer(self) -> str:
        """Get formatted answer with sources"""
        answer_with_sources = self.answer
        
        if self.sources:
            unique_sources = list(set(self.sources))
            sources_text = "\n\n**Sources:**\n"
            for i, source in enumerate(unique_sources, 1):
                sources_text += f"{i}. {source}\n"
            answer_with_sources += sources_text
        
        return answer_with_sources

class LanguageModelInterface(ABC):
    """Abstract interface for language models"""
    
    @abstractmethod
    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate response from prompt"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get model identifier"""
        pass

class OpenAILanguageModel(LanguageModelInterface):
    """OpenAI GPT model implementation"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI model
        
        Args:
            api_key: OpenAI API key
            model: Model name to use
        """
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            self.model = model
            self.available = True
            logger.info(f"Initialized OpenAI model: {model}")
        except ImportError:
            logger.error("OpenAI library not available. Install with: pip install openai")
            self.available = False
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI model: {e}")
            self.available = False
    
    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate response using OpenAI API"""
        if not self.available:
            return "OpenAI model not available"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def get_model_name(self) -> str:
        """Get model identifier"""
        return f"openai-{self.model}"

class MockLanguageModel(LanguageModelInterface):
    """Mock language model for testing without API calls"""
    
    def __init__(self):
        """Initialize mock model"""
        self.model_name = "mock-llm"
        logger.info("Initialized mock language model for testing")
    
    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate mock response based on context"""
        # Extract context from prompt
        if "Context:" in prompt and "Question:" in prompt:
            context_start = prompt.find("Context:") + 8
            question_start = prompt.find("Question:")
            
            if context_start < question_start:
                context = prompt[context_start:question_start].strip()
                question = prompt[question_start + 9:].strip()
                
                # Simple mock response generation
                if any(word in question.lower() for word in ["what", "define", "explain"]):
                    return f"Based on the provided context, {question.lower()} refers to concepts mentioned in the documentation. The context provides relevant information about this topic."
                elif any(word in question.lower() for word in ["how", "process", "method"]):
                    return f"According to the context provided, the process involves the steps and methods described in the documentation. Please refer to the source materials for detailed implementation."
                elif any(word in question.lower() for word in ["why", "reason", "purpose"]):
                    return f"The context explains that this is important because of the reasons outlined in the source documentation. The purpose is described in the provided materials."
                else:
                    return f"Based on the context provided, the answer to your question can be found in the referenced documentation. The information available suggests relevant details about your query."
        
        return "I can provide an answer based on the context provided. Please refer to the source materials for more detailed information."
    
    def get_model_name(self) -> str:
        """Get model identifier"""
        return self.model_name

class AnswerGenerator:
    """
    Answer generation system for RAG
    
    Handles prompt preparation, language model interaction, and response formatting
    with source attribution and context grounding.
    """
    
    def __init__(self, retriever: DocumentRetriever, settings: Optional[Settings] = None):
        """
        Initialize answer generator
        
        Args:
            retriever: Document retriever instance
            settings: Configuration settings
        """
        self.retriever = retriever
        self.settings = settings or Settings()
        
        # Initialize language model
        self.language_model = self._initialize_language_model()
        
        # Generation configuration
        self.max_context_length = getattr(self.settings, 'MAX_CONTEXT_LENGTH', 3000)
        self.max_answer_tokens = getattr(self.settings, 'MAX_ANSWER_TOKENS', 500)
        self.min_context_relevance = getattr(self.settings, 'MIN_CONTEXT_RELEVANCE', 0.3)
        
        # Performance tracking
        self.generation_stats = {
            "total_questions": 0,
            "successful_generations": 0,
            "total_generation_time": 0.0,
            "average_generation_time": 0.0
        }
        
        logger.info(f"Initialized answer generator with {self.language_model.get_model_name()}")
    
    def _initialize_language_model(self) -> LanguageModelInterface:
        """Initialize the appropriate language model"""
        # Try OpenAI first if API key is available
        openai_key = getattr(self.settings, 'OPENAI_API_KEY', None)
        if openai_key and openai_key.strip():
            try:
                model = OpenAILanguageModel(openai_key, "gpt-3.5-turbo")
                if model.available:
                    return model
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI model: {e}")
        
        # Fall back to mock model
        logger.info("Using mock language model (no API key or OpenAI unavailable)")
        return MockLanguageModel()
    
    def generate_answer(
        self,
        question: str,
        max_chunks: int = 5,
        include_sources: bool = True
    ) -> AnswerResult:
        """
        Generate answer for a question using retrieved context
        
        Args:
            question: User question
            max_chunks: Maximum number of context chunks to use
            include_sources: Whether to include source URLs
            
        Returns:
            AnswerResult with generated answer and metadata
        """
        start_time = time.time()
        
        logger.info(f"Generating answer for: '{question}'")
        
        try:
            # Step 1: Retrieve relevant context
            retrieval_result = self.retriever.retrieve(
                query=question,
                top_k=max_chunks,
                similarity_threshold=self.min_context_relevance
            )
            
            if retrieval_result.total_chunks == 0:
                logger.warning("No relevant context found for question")
                return self._create_no_context_answer(question, start_time)
            
            logger.info(f"Retrieved {retrieval_result.total_chunks} relevant chunks")
            
            # Step 2: Prepare context and prompt
            context_text = self._prepare_context(retrieval_result.chunks)
            prompt = self._create_prompt(question, context_text)
            
            # Step 3: Generate answer using language model
            generated_answer = self.language_model.generate_response(
                prompt, 
                max_tokens=self.max_answer_tokens
            )
            
            # Step 4: Post-process and validate answer
            processed_answer = self._post_process_answer(generated_answer, question)
            confidence_score = self._calculate_confidence(retrieval_result, processed_answer)
            
            generation_time = time.time() - start_time
            
            # Update statistics
            self.generation_stats["total_questions"] += 1
            self.generation_stats["successful_generations"] += 1
            self.generation_stats["total_generation_time"] += generation_time
            self.generation_stats["average_generation_time"] = (
                self.generation_stats["total_generation_time"] /
                self.generation_stats["successful_generations"]
            )
            
            result = AnswerResult(
                question=question,
                answer=processed_answer,
                sources=retrieval_result.get_unique_sources() if include_sources else [],
                context_used=context_text,
                confidence_score=confidence_score,
                generation_time=generation_time,
                model_used=self.language_model.get_model_name(),
                chunk_count=retrieval_result.total_chunks,
                retrieval_scores=retrieval_result.relevance_scores
            )
            
            logger.info(f"Generated answer in {generation_time:.3f}s (confidence: {confidence_score:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return self._create_error_answer(question, str(e), start_time)
    
    def _prepare_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Prepare context text from retrieved chunks
        
        Args:
            chunks: Retrieved document chunks
            
        Returns:
            Formatted context text
        """
        context_parts = []
        total_length = 0
        
        for i, chunk in enumerate(chunks):
            content = chunk.get('content', '').strip()
            source = chunk.get('metadata', {}).get('source', 'Unknown')
            
            # Add source information
            context_part = f"[Source {i+1}: {source}]\n{content}\n"
            
            # Check length limits
            if total_length + len(context_part) > self.max_context_length:
                logger.warning(f"Context length limit reached, using {i} of {len(chunks)} chunks")
                break
            
            context_parts.append(context_part)
            total_length += len(context_part)
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, question: str, context: str) -> str:
        """
        Create prompt for language model
        
        Args:
            question: User question
            context: Retrieved context
            
        Returns:
            Formatted prompt
        """
        prompt = f"""You are a helpful assistant that answers questions based only on the provided context. 

Instructions:
1. Answer the question using ONLY the information provided in the context below
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Do not make up information or use knowledge outside the provided context
4. Be concise but comprehensive in your answer
5. If you reference specific information, mention which source it comes from

Context:
{context}

Question: {question}

Answer:"""
        
        return prompt
    
    def _post_process_answer(self, raw_answer: str, question: str) -> str:
        """
        Post-process generated answer
        
        Args:
            raw_answer: Raw answer from language model
            question: Original question
            
        Returns:
            Cleaned and formatted answer
        """
        # Clean up the answer
        answer = raw_answer.strip()
        
        # Remove any potential prompt leakage
        if "Answer:" in answer:
            answer = answer.split("Answer:", 1)[1].strip()
        
        # Ensure answer is not empty
        if not answer or len(answer.strip()) < 10:
            answer = "I couldn't generate a complete answer based on the provided context. Please refer to the source materials for more information."
        
        return answer
    
    def _calculate_confidence(self, retrieval_result: RetrievalResult, answer: str) -> float:
        """
        Calculate confidence score for the generated answer
        
        Args:
            retrieval_result: Retrieval results
            answer: Generated answer
            
        Returns:
            Confidence score between 0 and 1
        """
        if not retrieval_result.relevance_scores:
            return 0.0
        
        # Base confidence on retrieval scores
        avg_retrieval_score = sum(retrieval_result.relevance_scores) / len(retrieval_result.relevance_scores)
        
        # Adjust based on answer quality indicators
        answer_quality = 1.0
        
        # Penalize very short answers
        if len(answer.split()) < 10:
            answer_quality *= 0.7
        
        # Penalize generic responses
        generic_phrases = ["based on the context", "according to the documentation", "please refer to"]
        generic_count = sum(1 for phrase in generic_phrases if phrase in answer.lower())
        if generic_count > 2:
            answer_quality *= 0.8
        
        # Bonus for specific information
        if any(word in answer.lower() for word in ["specifically", "exactly", "precisely", "details"]):
            answer_quality *= 1.1
        
        confidence = min(avg_retrieval_score * answer_quality, 1.0)
        return round(confidence, 3)
    
    def _create_no_context_answer(self, question: str, start_time: float) -> AnswerResult:
        """Create answer when no relevant context is found"""
        generation_time = time.time() - start_time
        
        return AnswerResult(
            question=question,
            answer="I couldn't find relevant information in the available documents to answer your question. Please try rephrasing your question or check if the topic is covered in the crawled content.",
            sources=[],
            context_used="",
            confidence_score=0.0,
            generation_time=generation_time,
            model_used=self.language_model.get_model_name(),
            chunk_count=0,
            retrieval_scores=[]
        )
    
    def _create_error_answer(self, question: str, error: str, start_time: float) -> AnswerResult:
        """Create answer when generation fails"""
        generation_time = time.time() - start_time
        
        return AnswerResult(
            question=question,
            answer=f"I encountered an error while generating an answer: {error}. Please try again.",
            sources=[],
            context_used="",
            confidence_score=0.0,
            generation_time=generation_time,
            model_used=self.language_model.get_model_name(),
            chunk_count=0,
            retrieval_scores=[]
        )
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get answer generation statistics"""
        retrieval_stats = self.retriever.get_retrieval_stats()
        
        return {
            "generation": self.generation_stats,
            "retrieval": retrieval_stats,
            "model": {
                "name": self.language_model.get_model_name(),
                "max_context_length": self.max_context_length,
                "max_answer_tokens": self.max_answer_tokens
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on answer generation system
        
        Returns:
            Health status information
        """
        health = {
            "retriever_ready": False,
            "language_model_ready": False,
            "test_generation_success": False
        }
        
        try:
            # Check retriever
            retriever_health = self.retriever.health_check()
            health["retriever_ready"] = retriever_health.get("test_query_success", False)
            
            # Check language model
            test_response = self.language_model.generate_response("Test prompt", max_tokens=10)
            health["language_model_ready"] = bool(test_response and len(test_response.strip()) > 0)
            
            # Test full generation pipeline
            if health["retriever_ready"] and health["language_model_ready"]:
                test_result = self.generate_answer("test question")
                health["test_generation_success"] = test_result.confidence_score >= 0.0
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health["error"] = str(e)
        
        return health