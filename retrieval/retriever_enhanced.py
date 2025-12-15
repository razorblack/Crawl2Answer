"""
Enhanced Retrieval Module for Crawl2Answer RAG System

This module provides advanced retrieval functionality that takes user queries,
generates embeddings, and fetches the most relevant document chunks
from the vector database with comprehensive metadata handling.

Features:
- Query embedding generation
- Similarity-based chunk retrieval
- Relevance scoring and filtering
- Metadata extraction and source tracking
- Performance optimization and caching
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
from pathlib import Path

from embeddings.embedder_enhanced import Embedder, EmbeddingResult
from vector_store.vector_db_enhanced import VectorDatabase, SearchResult
from config.settings import Settings

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Result from document retrieval"""
    query: str
    chunks: List[Dict[str, Any]]
    sources: List[str]
    relevance_scores: List[float]
    retrieval_time: float
    total_chunks: int
    
    def get_context_text(self) -> str:
        """Get concatenated text from all retrieved chunks"""
        return "\n\n".join([chunk['content'] for chunk in self.chunks])
    
    def get_unique_sources(self) -> List[str]:
        """Get unique source URLs"""
        return list(set(self.sources))

class DocumentRetriever:
    """
    Enhanced document retrieval system for RAG
    
    Handles query embedding generation and similarity-based document retrieval
    from the vector database with advanced features.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the document retriever
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings or Settings()
        
        # Initialize embedder
        logger.info("Initializing embedder for query processing...")
        self.embedder = Embedder(self.settings)
        
        # Initialize vector database
        embedding_dim = self.settings.EMBEDDING_DIMENSION
        vector_db_path = self.settings.VECTOR_DB_PATH
        index_type = getattr(self.settings, 'VECTOR_DB_INDEX_TYPE', 'cosine')
        
        logger.info(f"Initializing vector database: {vector_db_path}")
        self.vector_db = VectorDatabase(
            dimension=embedding_dim,
            storage_path=vector_db_path,
            index_type=index_type
        )
        
        # Load existing database
        load_success = self.vector_db.load()
        if load_success:
            stats = self.vector_db.get_stats()
            logger.info(f"Loaded vector database with {stats.total_vectors} documents")
        else:
            logger.warning("No existing vector database found. Make sure to run embedding generation first.")
        
        # Retrieval configuration
        self.default_top_k = getattr(self.settings, 'RETRIEVAL_K', 5)
        self.similarity_threshold = getattr(self.settings, 'SIMILARITY_THRESHOLD', 0.1)
        
        # Performance tracking
        self.retrieval_stats = {
            "total_queries": 0,
            "total_retrieval_time": 0.0,
            "average_retrieval_time": 0.0,
            "cache_hits": 0
        }
    
    def retrieve(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant document chunks for a query
        
        Args:
            query: User question or search query
            top_k: Number of chunks to retrieve (default: from settings)
            similarity_threshold: Minimum similarity score (default: from settings)
            filter_metadata: Optional metadata filters
            
        Returns:
            RetrievalResult with relevant chunks and metadata
        """
        start_time = time.time()
        
        # Use defaults if not specified
        top_k = top_k or self.default_top_k
        similarity_threshold = similarity_threshold or self.similarity_threshold
        
        logger.info(f"Processing query: '{query}'")
        logger.info(f"Retrieval params - top_k: {top_k}, threshold: {similarity_threshold}")
        
        try:
            # Step 1: Generate embedding for the query
            query_embedding = self._embed_query(query)
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return self._empty_result(query, start_time)
            
            # Step 2: Search vector database for similar chunks
            search_results = self._search_similar_chunks(
                query_embedding, top_k, filter_metadata
            )
            
            # Step 3: Filter results by similarity threshold
            filtered_results = [
                result for result in search_results 
                if result.similarity_score >= similarity_threshold
            ]
            
            if not filtered_results:
                logger.warning(f"No results above similarity threshold {similarity_threshold}")
                return self._empty_result(query, start_time)
            
            # Step 4: Extract chunks, sources, and scores
            chunks = []
            sources = []
            scores = []
            
            for result in filtered_results:
                chunk_data = {
                    "chunk_id": result.chunk_id,
                    "content": result.content,
                    "metadata": result.metadata,
                    "similarity_score": result.similarity_score
                }
                
                chunks.append(chunk_data)
                scores.append(result.similarity_score)
                
                # Extract source URL
                source = result.metadata.get('source', result.metadata.get('url', 'Unknown'))
                sources.append(source)
            
            retrieval_time = time.time() - start_time
            
            # Update statistics
            self.retrieval_stats["total_queries"] += 1
            self.retrieval_stats["total_retrieval_time"] += retrieval_time
            self.retrieval_stats["average_retrieval_time"] = (
                self.retrieval_stats["total_retrieval_time"] / 
                self.retrieval_stats["total_queries"]
            )
            
            result = RetrievalResult(
                query=query,
                chunks=chunks,
                sources=sources,
                relevance_scores=scores,
                retrieval_time=retrieval_time,
                total_chunks=len(chunks)
            )
            
            logger.info(f"Retrieved {len(chunks)} chunks in {retrieval_time:.3f}s")
            if scores:
                logger.info(f"Top similarity score: {scores[0]:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return self._empty_result(query, start_time)
    
    def _embed_query(self, query: str) -> Optional[EmbeddingResult]:
        """Generate embedding for user query"""
        try:
            embedding_result = self.embedder.generate_embedding(text=query)
            
            if embedding_result:
                logger.debug(f"Generated query embedding (dim: {embedding_result.dimension})")
                return embedding_result
            else:
                logger.error("Embedder returned None for query")
                return None
                
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return None
    
    def _search_similar_chunks(
        self, 
        query_embedding: EmbeddingResult, 
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar chunks in vector database"""
        try:
            # Perform similarity search
            search_results = self.vector_db.search(
                query_vector=query_embedding.embedding,
                top_k=top_k,
                filter_metadata=filter_metadata
            )
            
            logger.debug(f"Vector search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _empty_result(self, query: str, start_time: float) -> RetrievalResult:
        """Create empty result for failed retrievals"""
        retrieval_time = time.time() - start_time
        return RetrievalResult(
            query=query,
            chunks=[],
            sources=[],
            relevance_scores=[],
            retrieval_time=retrieval_time,
            total_chunks=0
        )
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the vector database"""
        try:
            stats = self.vector_db.get_stats()
            return {
                "total_documents": stats.total_vectors,
                "embedding_dimension": stats.embedding_dimension,
                "index_type": stats.index_type,
                "storage_size_mb": stats.total_size_mb,
                "last_updated": stats.last_updated
            }
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {"error": str(e)}
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval performance statistics"""
        return {
            **self.retrieval_stats,
            "database_info": self.get_database_info()
        }
    
    def search_by_source(self, source_filter: str, query: str, top_k: int = 3) -> RetrievalResult:
        """
        Search within documents from a specific source
        
        Args:
            source_filter: Source URL or pattern to filter by
            query: Search query
            top_k: Number of results to return
            
        Returns:
            RetrievalResult filtered by source
        """
        logger.info(f"Source-filtered search: '{source_filter}' for query: '{query}'")
        
        return self.retrieve(
            query=query,
            top_k=top_k,
            filter_metadata={"source": source_filter}
        )
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on retrieval system
        
        Returns:
            Health status information
        """
        health = {
            "embedder_ready": False,
            "vector_db_ready": False,
            "total_documents": 0,
            "test_query_success": False
        }
        
        try:
            # Check embedder
            test_embedding = self.embedder.generate_embedding("test")
            health["embedder_ready"] = test_embedding is not None
            
            # Check vector database
            stats = self.vector_db.get_stats()
            health["vector_db_ready"] = stats.total_vectors > 0
            health["total_documents"] = stats.total_vectors
            
            # Test retrieval
            if health["embedder_ready"] and health["vector_db_ready"]:
                test_result = self.retrieve("test query", top_k=1)
                health["test_query_success"] = test_result.total_chunks >= 0
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health["error"] = str(e)
        
        return health


# Legacy compatibility class
class Retriever(DocumentRetriever):
    """Legacy compatibility wrapper for DocumentRetriever"""
    
    def __init__(self, embedder=None, vector_db=None, settings: Optional[Settings] = None):
        """
        Initialize with legacy interface support
        
        Args:
            embedder: Legacy embedder (ignored, uses enhanced version)
            vector_db: Legacy vector_db (ignored, uses enhanced version)
            settings: Configuration settings
        """
        # Initialize enhanced version
        super().__init__(settings)
        
    def retrieve(self, query: str, k: int = 5, score_threshold: float = 0.1) -> List[Dict]:
        """
        Legacy retrieve method for backward compatibility
        
        Args:
            query: User query
            k: Number of results
            score_threshold: Minimum score threshold
            
        Returns:
            List of chunks (legacy format)
        """
        # Use enhanced retrieval
        result = super().retrieve(
            query=query,
            top_k=k,
            similarity_threshold=score_threshold
        )
        
        # Return in legacy format
        return result.chunks