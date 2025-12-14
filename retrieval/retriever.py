"""
Retrieval module for finding relevant content based on queries.
"""

from typing import List, Dict, Optional
import logging
from embeddings.embedder import Embedder
from vector_store.vector_db import VectorDatabase

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieve relevant content chunks for a given query."""
    
    def __init__(self, embedder: Embedder, vector_db: VectorDatabase):
        """
        Initialize the retriever.
        
        Args:
            embedder: Embedder instance for generating query embeddings
            vector_db: Vector database for similarity search
        """
        self.embedder = embedder
        self.vector_db = vector_db
    
    def retrieve(self, query: str, k: int = 5, 
                score_threshold: float = 0.1) -> List[Dict]:
        """
        Retrieve relevant content chunks for a query.
        
        Args:
            query: User query
            k: Number of top results to retrieve
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of relevant chunks with metadata and scores
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedder.generate_embedding(query.strip())
            if query_embedding is None:
                logger.error("Failed to generate query embedding")
                return []
            
            # Search vector database
            results = self.vector_db.search(
                query_embedding, 
                k=k, 
                score_threshold=score_threshold
            )
            
            # Post-process results
            processed_results = self._post_process_results(results, query)
            
            logger.info(f"Retrieved {len(processed_results)} relevant chunks for query: {query[:50]}...")
            return processed_results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    def _post_process_results(self, results: List[Dict], query: str) -> List[Dict]:
        """
        Post-process search results to improve relevance.
        
        Args:
            results: Raw search results
            query: Original query
            
        Returns:
            Processed results
        """
        if not results:
            return results
        
        # Add query context
        for result in results:
            result['query'] = query
            result['retrieval_timestamp'] = self._get_timestamp()
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        return results
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def retrieve_with_context(self, query: str, k: int = 5, 
                             context_window: int = 2) -> List[Dict]:
        """
        Retrieve relevant chunks with additional context from neighboring chunks.
        
        Args:
            query: User query
            k: Number of top results to retrieve
            context_window: Number of neighboring chunks to include as context
            
        Returns:
            List of relevant chunks with expanded context
        """
        # Get initial results
        results = self.retrieve(query, k)
        
        # This would require more sophisticated chunk tracking
        # For now, return regular results
        # TODO: Implement context window expansion
        
        return results
    
    def get_retrieval_stats(self) -> Dict:
        """
        Get retrieval statistics.
        
        Returns:
            Dictionary with retrieval statistics
        """
        db_stats = self.vector_db.get_stats()
        embedder_info = {
            'model_type': self.embedder.model_type,
            'model_name': self.embedder.model_name,
            'embedding_dimension': self.embedder.get_embedding_dimension()
        }
        
        return {
            'vector_database': db_stats,
            'embedder': embedder_info
        }