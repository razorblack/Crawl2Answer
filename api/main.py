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
    max_depth: int = 3
    delay: float = 1.0


class CrawlResponse(BaseModel):
    status: str
    message: str
    crawled_pages: int
    base_domain: str
    total_content_size: int
    pages: List[Dict]


class QuestionRequest(BaseModel):
    question: str
    max_results: int = 5


class TextExtractionResponse(BaseModel):
    status: str
    message: str
    url: str
    title: str
    word_count: int
    char_count: int
    cleaned_text_preview: str
    metadata: Dict


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict]
    confidence: float
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


@app.post("/crawl", response_model=CrawlResponse)
async def crawl_website(request: CrawlRequest):
    """Crawl a website and add content to the knowledge base."""
    try:
        # Initialize crawler with enhanced settings
        crawler = WebCrawler(
            base_url=request.url,
            delay=request.delay,
            max_depth=request.max_depth
        )
        text_extractor = TextExtractor()
        chunker = TextChunker()
        embedder = Embedder(
            model_type=settings.EMBEDDING_MODEL_TYPE,
            model_name=settings.EMBEDDING_MODEL_NAME
        )
        
        # Crawl website with enhanced functionality
        crawled_pages = crawler.crawl_site(max_pages=request.max_pages)
        if not crawled_pages:
            raise HTTPException(status_code=400, detail="Failed to crawl any pages")
        
        # Process each page with enhanced extraction
        all_chunks = []
        for page in crawled_pages:
            # Extract and clean text using enhanced extractor
            cleaned_content = text_extractor.extract_text(
                html_content=page.html_content,
                url=page.url,
                title=page.title
            )
            
            if not cleaned_content:
                logger.warning(f"Failed to extract text from {page.url}")
                continue
            
            # Use cleaned text and enhanced metadata
            metadata = cleaned_content.metadata
            metadata.update({
                'source_url': page.url,
                'page_title': cleaned_content.title,
                'crawl_timestamp': page.crawl_timestamp,
                'status_code': page.status_code,
                'extraction_timestamp': cleaned_content.extraction_timestamp,
                'word_count': cleaned_content.word_count,
                'char_count': cleaned_content.char_count
            })
            
            # Chunk the cleaned text
            chunks = chunker.chunk_text(cleaned_content.cleaned_text, metadata)
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
        
        # Get crawl statistics
        stats = crawler.get_crawl_stats(crawled_pages)
        
        # Prepare response
        page_summaries = [
            {
                'url': page.url,
                'title': page.title,
                'status_code': page.status_code,
                'content_size': len(page.html_content),
                'timestamp': page.crawl_timestamp
            }
            for page in crawled_pages
        ]
        
        return CrawlResponse(
            status="success",
            message=f"Successfully crawled {len(crawled_pages)} pages from {stats['base_domain']}",
            crawled_pages=len(crawled_pages),
            base_domain=stats['base_domain'],
            total_content_size=stats['total_content_size'],
            pages=page_summaries
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


@app.post("/test-crawl", response_model=CrawlResponse)
async def test_crawl_website(request: CrawlRequest):
    """Test crawl a website and return crawled URLs without processing content."""
    try:
        # Initialize crawler
        crawler = WebCrawler(
            base_url=request.url,
            delay=request.delay,
            max_depth=request.max_depth
        )
        
        # Crawl website
        crawled_pages = crawler.crawl_site(max_pages=request.max_pages)
        if not crawled_pages:
            raise HTTPException(status_code=400, detail="Failed to crawl any pages")
        
        # Get crawl statistics
        stats = crawler.get_crawl_stats(crawled_pages)
        
        # Prepare response with page summaries
        page_summaries = [
            {
                'url': page.url,
                'title': page.title,
                'status_code': page.status_code,
                'content_size': len(page.html_content),
                'timestamp': page.crawl_timestamp
            }
            for page in crawled_pages
        ]
        
        return CrawlResponse(
            status="success",
            message=f"Test crawl completed. Found {len(crawled_pages)} pages on {stats['base_domain']}",
            crawled_pages=len(crawled_pages),
            base_domain=stats['base_domain'],
            total_content_size=stats['total_content_size'],
            pages=page_summaries
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Test crawling failed: {e}")
        raise HTTPException(status_code=500, detail=f"Test crawling failed: {str(e)}")


@app.post("/test-extraction", response_model=TextExtractionResponse)
async def test_text_extraction(request: CrawlRequest):
    """Test text extraction from a single page without storing in database."""
    try:
        # Initialize components
        crawler = WebCrawler(
            base_url=request.url,
            delay=request.delay,
            max_depth=1  # Only fetch the single page
        )
        text_extractor = TextExtractor()
        
        # Fetch just the single page
        page = crawler.fetch_page(request.url)
        if not page:
            raise HTTPException(status_code=400, detail="Failed to fetch the page")
        
        # Extract and clean text
        cleaned_content = text_extractor.extract_text(
            html_content=page.html_content,
            url=page.url,
            title=page.title
        )
        
        if not cleaned_content:
            raise HTTPException(status_code=400, detail="Failed to extract text from page")
        
        # Log the cleaned content for testing
        text_extractor.log_cleaned_content(cleaned_content)
        
        # Prepare preview (first 500 characters)
        preview_length = 500
        text_preview = cleaned_content.cleaned_text
        if len(text_preview) > preview_length:
            text_preview = text_preview[:preview_length] + "..."
        
        return TextExtractionResponse(
            status="success",
            message=f"Successfully extracted text from {page.url}",
            url=cleaned_content.url,
            title=cleaned_content.title,
            word_count=cleaned_content.word_count,
            char_count=cleaned_content.char_count,
            cleaned_text_preview=text_preview,
            metadata=cleaned_content.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text extraction test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Text extraction test failed: {str(e)}")


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