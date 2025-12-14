"""
FastAPI application for the Crawl2Answer Q&A bot.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import logging
import os
from pathlib import Path

# Import our modules
from crawling.crawler import WebCrawler
from extraction.text_extractor import TextExtractor
from chunking.chunker import TextChunker
from embeddings.embedder import Embedder
from vector_store.vector_db import VectorDatabase
from retrieval.retriever import Retriever
from config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Crawl2Answer API",
    description="Q&A bot using Retrieval Augmented Generation (RAG)",
    version="1.0.0"
)

# Global components (will be initialized on startup)
settings = None
retriever = None


# Pydantic models for API requests/responses
class CrawlRequest(BaseModel):
    url: str
    max_pages: int = 10


class QuestionRequest(BaseModel):
    question: str
    max_results: int = 5


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict]
    confidence: float


class StatusResponse(BaseModel):
    status: str
    message: str
    stats: Optional[Dict] = None


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global settings, retriever
    
    try:
        # Load settings
        settings = Settings()
        
        # Initialize embedder
        embedder = Embedder(
            model_type=settings.EMBEDDING_MODEL_TYPE,
            model_name=settings.EMBEDDING_MODEL_NAME
        )
        
        # Initialize vector database
        vector_db = VectorDatabase(
            dimension=embedder.get_embedding_dimension(),
            storage_path=settings.VECTOR_DB_PATH
        )
        
        # Load existing database
        vector_db.load()
        
        # Initialize retriever
        retriever = Retriever(embedder, vector_db)
        
        logger.info("Crawl2Answer API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise


@app.get("/", response_model=StatusResponse)
async def root():
    """Root endpoint with API information."""
    return StatusResponse(
        status="running",
        message="Crawl2Answer API is running",
        stats=retriever.get_retrieval_stats() if retriever else None
    )


@app.post("/crawl", response_model=StatusResponse)
async def crawl_website(request: CrawlRequest):
    """Crawl a website and add content to the knowledge base."""
    try:
        # Initialize components
        crawler = WebCrawler(request.url)
        text_extractor = TextExtractor()
        chunker = TextChunker()
        embedder = Embedder(
            model_type=settings.EMBEDDING_MODEL_TYPE,
            model_name=settings.EMBEDDING_MODEL_NAME
        )
        
        # Crawl website
        crawled_pages = crawler.crawl_site(max_pages=request.max_pages)
        if not crawled_pages:
            raise HTTPException(status_code=400, detail="Failed to crawl any pages")
        
        # Process each page
        all_chunks = []
        for url, html_content in crawled_pages:
            # Extract text
            text = text_extractor.extract_text(html_content)
            if not text:
                continue
            
            # Extract metadata
            metadata = text_extractor.extract_metadata(html_content)
            metadata['source_url'] = url
            
            # Chunk text
            chunks = chunker.chunk_text(text, metadata)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            raise HTTPException(status_code=400, detail="No text content extracted")
        
        # Generate embeddings
        chunks_with_embeddings = embedder.embed_chunks(all_chunks)
        
        # Add to vector database
        success = retriever.vector_db.add_embeddings(chunks_with_embeddings)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to add embeddings to database")
        
        # Save database
        retriever.vector_db.save()
        
        return StatusResponse(
            status="success",
            message=f"Successfully crawled and processed {len(crawled_pages)} pages, created {len(chunks_with_embeddings)} chunks",
            stats={
                'pages_crawled': len(crawled_pages),
                'chunks_created': len(chunks_with_embeddings),
                'total_chunks_in_db': retriever.vector_db.get_stats()['total_vectors']
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Crawling failed: {e}")
        raise HTTPException(status_code=500, detail=f"Crawling failed: {str(e)}")


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question and get an answer based on retrieved content."""
    try:
        # Retrieve relevant content
        relevant_chunks = retriever.retrieve(
            request.question, 
            k=request.max_results
        )
        
        if not relevant_chunks:
            raise HTTPException(
                status_code=404, 
                detail="No relevant content found for this question"
            )
        
        # Generate answer from retrieved content
        answer = _generate_answer(request.question, relevant_chunks)
        
        # Calculate confidence (average similarity score)
        confidence = sum(chunk.get('similarity_score', 0) for chunk in relevant_chunks) / len(relevant_chunks)
        
        # Prepare sources
        sources = [
            {
                'content': chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content'],
                'source_url': chunk.get('metadata', {}).get('source_url', 'unknown'),
                'similarity_score': chunk.get('similarity_score', 0)
            }
            for chunk in relevant_chunks
        ]
        
        return AnswerResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            confidence=confidence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Question answering failed: {e}")
        raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get system status and statistics."""
    try:
        stats = retriever.get_retrieval_stats() if retriever else {}
        
        return StatusResponse(
            status="healthy",
            message="System is operational",
            stats=stats
        )
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@app.delete("/clear", response_model=StatusResponse)
async def clear_database():
    """Clear all data from the knowledge base."""
    try:
        success = retriever.vector_db.clear()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to clear database")
        
        return StatusResponse(
            status="success",
            message="Knowledge base cleared successfully"
        )
        
    except Exception as e:
        logger.error(f"Database clearing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database clearing failed: {str(e)}")


def _generate_answer(question: str, relevant_chunks: List[Dict]) -> str:
    """
    Generate an answer based on retrieved content.
    This is a simple implementation that concatenates relevant content.
    In a real implementation, you might use a language model to generate more coherent answers.
    
    Args:
        question: The user's question
        relevant_chunks: List of relevant content chunks
        
    Returns:
        Generated answer string
    """
    if not relevant_chunks:
        return "I couldn't find any relevant information to answer your question."
    
    # Simple approach: concatenate relevant content
    answer_parts = []
    answer_parts.append("Based on the available information:")
    answer_parts.append("")
    
    for i, chunk in enumerate(relevant_chunks[:3], 1):  # Use top 3 chunks
        content = chunk['content']
        source_url = chunk.get('metadata', {}).get('source_url', 'unknown source')
        
        # Truncate very long content
        if len(content) > 500:
            content = content[:497] + "..."
        
        answer_parts.append(f"{i}. {content}")
        answer_parts.append(f"   (Source: {source_url})")
        answer_parts.append("")
    
    return "\n".join(answer_parts)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)