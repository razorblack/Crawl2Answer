"""
FastAPI application for the Crawl2Answer Q&A bot.
Step 7: REST API endpoints using enhanced components from Steps 1-6.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import os
import asyncio
import time
from pathlib import Path

# Import our enhanced modules
from crawling.crawler import WebCrawler
from extraction.text_extractor import TextExtractor  
from chunking.chunker import TextChunker
from embeddings.embedder_enhanced import Embedder
from vector_store.vector_db import VectorDatabase
from retrieval.retriever_enhanced import DocumentRetriever
from generation.answer_generator import AnswerGenerator
from config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Crawl2Answer API",
    description="Advanced Q&A bot using Retrieval Augmented Generation (RAG) with Steps 1-6 integration",
    version="2.0.0"
)

# Global components (will be initialized on startup)
settings = None
retriever = None
answer_generator = None


# Pydantic models for API requests/responses
class CrawlRequest(BaseModel):
    baseUrl: str  # Changed from 'url' to match Step 7 requirements
    max_pages: int = 10
    max_depth: int = 3
    delay: float = 1.0


class CrawlResponse(BaseModel):
    status: str
    message: str
    pages_crawled: int
    chunks_created: int
    embeddings_generated: int
    database_updated: bool
    processing_time: float
    base_url: str


class QuestionRequest(BaseModel):
    question: str  # Main input for /ask endpoint


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]  # List of source URLs as required
    confidence: float
    retrieval_time: float
    generation_time: float


class StatusResponse(BaseModel):
    status: str
    message: str
    stats: Optional[Dict] = None


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global settings, retriever, answer_generator
    
    try:
        # Load settings
        settings = Settings()
        logger.info("Settings loaded successfully")
        
        # Initialize enhanced components
        retriever = DocumentRetriever(settings)
        answer_generator = AnswerGenerator(retriever, settings)
        
        # Check system health
        retriever_health = retriever.health_check()
        generator_health = answer_generator.health_check()
        
        logger.info(f"Retriever health: {retriever_health}")
        logger.info(f"Answer generator health: {generator_health}")
        logger.info("Crawl2Answer API started successfully with enhanced Step 6 components")
        
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
    """
    Step 7 Implementation: POST /crawl endpoint
    
    Actions:
    1. Run crawling
    2. Run extraction  
    3. Run chunking
    4. Run embeddings
    5. Index everything in the vector store
    
    Output: Success message with processing statistics
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting crawl process for: {request.baseUrl}")
        
        # Step 1: Initialize crawler  
        crawler = WebCrawler(
            base_url=request.baseUrl,
            max_pages=request.max_pages,
            max_depth=request.max_depth,
            delay=request.delay
        )
        
        # Step 2: Crawl website
        logger.info("Step 1: Running crawling...")
        crawled_pages = await asyncio.to_thread(
            crawler.crawl_site, 
            max_pages=request.max_pages
        )
        
        if not crawled_pages:
            raise HTTPException(status_code=400, detail="Failed to crawl any pages")
        
        logger.info(f"Crawled {len(crawled_pages)} pages")
        
        # Step 3: Initialize text extractor
        text_extractor = TextExtractor()
        
        # Step 4: Extract and chunk text
        logger.info("Step 2: Running extraction...")
        logger.info("Step 3: Running chunking...")
        
        all_chunks = []
        chunker = TextChunker()
        
        for page_data in crawled_pages:
            # Extract clean text
            extracted_content = await asyncio.to_thread(
                text_extractor.extract_text,
                page_data.html_content,
                page_data.url,
                page_data.title
            )
            
            if not extracted_content:
                logger.warning(f"Failed to extract text from {page_data.url}")
                continue
            
            # Prepare metadata
            metadata = {
                'source_url': page_data.url,
                'title': extracted_content.title,
                'crawl_timestamp': page_data.crawl_timestamp,
                'status_code': page_data.status_code,
                **extracted_content.metadata
            }
            
            # Generate chunks
            chunks = await asyncio.to_thread(
                chunker.chunk_text,
                extracted_content.cleaned_text,
                metadata
            )
            
            all_chunks.extend(chunks)
        
        if not all_chunks:
            raise HTTPException(status_code=400, detail="No text chunks generated")
        
        logger.info(f"Generated {len(all_chunks)} text chunks")
        
        # Step 5: Generate embeddings and index in vector store
        logger.info("Step 4: Running embeddings...")
        embedder = Embedder(settings)
        
        # Generate embeddings for chunks
        embeddings_results = []
        for chunk in all_chunks:
            embedding_result = await asyncio.to_thread(
                embedder.generate_embedding,
                chunk.content
            )
            
            if embedding_result:
                embeddings_results.append({
                    'embedding': embedding_result.embedding,
                    'chunk_id': chunk.chunk_id,
                    'content': chunk.content,
                    'metadata': chunk.metadata
                })
        
        if not embeddings_results:
            raise HTTPException(status_code=500, detail="Failed to generate embeddings")
        
        logger.info(f"Generated {len(embeddings_results)} embeddings")
        
        # Step 6: Index in vector store
        logger.info("Step 5: Indexing in vector store...")
        
        # Add embeddings to vector database through retriever
        vector_db = retriever.vector_db
        
        for embedding_data in embeddings_results:
            success = vector_db.add_embedding(
                embedding=embedding_data['embedding'],
                metadata=embedding_data['metadata'],
                content=embedding_data['content']
            )
            
            if not success:
                logger.warning(f"Failed to add embedding for chunk {embedding_data['chunk_id']}")
        
        # Save the updated database
        save_success = vector_db.save()
        
        processing_time = time.time() - start_time
        
        logger.info(f"Crawl process completed in {processing_time:.2f} seconds")
        
        return CrawlResponse(
            status="success",
            message=f"Successfully processed {len(crawled_pages)} pages from {request.baseUrl}",
            pages_crawled=len(crawled_pages),
            chunks_created=len(all_chunks),
            embeddings_generated=len(embeddings_results),
            database_updated=save_success,
            processing_time=processing_time,
            base_url=request.baseUrl
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Crawl process failed: {e}")
        raise HTTPException(status_code=500, detail=f"Crawl process failed: {str(e)}")


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Step 7 Implementation: POST /ask endpoint
    
    Actions:
    1. Run retrieval
    2. Generate final answer
    
    Output: answer text and source URLs
    """
    start_retrieval = time.time()
    
    try:
        logger.info(f"Processing question: {request.question}")
        
        # Step 1: Run retrieval
        logger.info("Running retrieval...")
        retrieval_result = await asyncio.to_thread(
            retriever.retrieve,
            request.question,
            top_k=5,
            similarity_threshold=0.1
        )
        
        retrieval_time = time.time() - start_retrieval
        
        if retrieval_result.total_chunks == 0:
            raise HTTPException(
                status_code=404,
                detail="No relevant content found for this question"
            )
        
        logger.info(f"Retrieved {retrieval_result.total_chunks} relevant chunks")
        
        # Step 2: Generate final answer
        logger.info("Generating answer...")
        start_generation = time.time()
        
        answer_result = await asyncio.to_thread(
            answer_generator.generate_answer,
            request.question
        )
        
        generation_time = time.time() - start_generation
        
        if not answer_result or not answer_result.answer:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate answer"
            )
        
        # Extract unique source URLs
        source_urls = list(set(answer_result.sources))
        
        logger.info(f"Generated answer with confidence {answer_result.confidence:.3f}")
        
        return AnswerResponse(
            question=request.question,
            answer=answer_result.answer,
            sources=source_urls,
            confidence=answer_result.confidence,
            retrieval_time=retrieval_time,
            generation_time=generation_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Question answering failed: {e}")
        raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")


@app.get("/", response_model=StatusResponse)
async def root():
    """Root endpoint with API information."""
    try:
        # Get system health information
        retriever_health = retriever.health_check() if retriever else {}
        generator_health = answer_generator.health_check() if answer_generator else {}
        
        stats = {
            "retriever": retriever_health,
            "generator": generator_health,
            "database_info": retriever.get_database_info() if retriever else {}
        }
        
        return StatusResponse(
            status="running",
            message="Crawl2Answer API v2.0 is running with enhanced Step 6 components",
            stats=stats
        )
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "message": "API is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)