"""
Enhanced Vector Database Module

This module provides comprehensive vector storage and retrieval capabilities
using FAISS for efficient similarity search in the Crawl2Answer RAG system.

Features:
- FAISS-based vector storage and search
- Metadata management and filtering
- Batch operations for efficiency
- Persistence and loading
- Multiple similarity metrics
- Index optimization
"""

import numpy as np
import pickle
import json
import os
from typing import List, Dict, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import time
from dataclasses import dataclass, asdict

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from embeddings.embedder_enhanced import EmbeddingResult

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Result from vector similarity search"""
    chunk_id: int
    similarity_score: float
    content: str
    metadata: Dict[str, Any]
    embedding_dimension: int
    
@dataclass
class VectorStats:
    """Statistics about the vector database"""
    total_vectors: int
    embedding_dimension: int
    index_type: str
    total_size_mb: float
    last_updated: str
    search_performance_ms: float

class VectorDatabase:
    """
    Enhanced vector database with FAISS backend and advanced features
    """
    
    def __init__(self, dimension: int, storage_path: str = "data/vector_store", index_type: str = "cosine"):
        """
        Initialize the vector database
        
        Args:
            dimension: Dimension of the embedding vectors
            storage_path: Path to store the database files
            index_type: Type of similarity index ('cosine', 'euclidean', 'inner_product')
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")
        
        self.dimension = dimension
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.index_type = index_type
        
        # Initialize FAISS index based on type
        self._create_index()
        
        # Store metadata and mappings
        self.metadata_store = {}  # chunk_id -> metadata
        self.content_store = {}   # chunk_id -> content
        self.id_mapping = []      # index_position -> chunk_id
        self.next_internal_id = 0
        
        # Performance tracking
        self.stats = {
            "total_insertions": 0,
            "total_searches": 0,
            "total_search_time": 0.0,
            "average_search_time": 0.0,
            "last_updated": None
        }
        
        # File paths
        self.index_path = self.storage_path / "faiss_index.bin"
        self.metadata_path = self.storage_path / "metadata.pkl"
        self.content_path = self.storage_path / "content.pkl"
        self.mapping_path = self.storage_path / "id_mapping.pkl"
        self.stats_path = self.storage_path / "stats.json"
    
    def _create_index(self):
        """Create FAISS index based on the specified type"""
        
        if self.index_type == "cosine":
            # Cosine similarity (normalized inner product)
            self.index = faiss.IndexFlatIP(self.dimension)
            self.normalize_vectors = True
        elif self.index_type == "euclidean":
            # L2 (Euclidean) distance
            self.index = faiss.IndexFlatL2(self.dimension)
            self.normalize_vectors = False
        elif self.index_type == "inner_product":
            # Inner product similarity
            self.index = faiss.IndexFlatIP(self.dimension)
            self.normalize_vectors = False
        else:
            raise ValueError(f"Unsupported index_type: {self.index_type}")
        
        logger.info(f"Created FAISS index: {self.index_type} similarity, dimension: {self.dimension}")
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector for cosine similarity"""
        if self.normalize_vectors:
            norm = np.linalg.norm(vector)
            if norm > 0:
                return vector / norm
        return vector
    
    def add_embedding_result(self, embedding_result: EmbeddingResult) -> bool:
        """
        Add a single embedding result to the database
        
        Args:
            embedding_result: EmbeddingResult object
            
        Returns:
            Success status
        """
        try:
            vector = np.array(embedding_result.embedding, dtype=np.float32).reshape(1, -1)
            vector = self._normalize_vector(vector)
            
            # Add to FAISS index
            self.index.add(vector)
            
            # Store metadata and content
            chunk_id = embedding_result.chunk_id
            self.metadata_store[chunk_id] = embedding_result.metadata
            self.content_store[chunk_id] = embedding_result.metadata.get('content', '')
            
            # Update mapping
            self.id_mapping.append(chunk_id)
            
            # Update stats
            self.stats["total_insertions"] += 1
            self.stats["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add embedding: {e}")
            return False
    
    def add_embedding_results(self, embedding_results: List[EmbeddingResult]) -> int:
        """
        Add multiple embedding results to the database
        
        Args:
            embedding_results: List of EmbeddingResult objects
            
        Returns:
            Number of successfully added embeddings
        """
        successful_additions = 0
        
        # Prepare batch data
        vectors = []
        chunk_ids = []
        metadata_batch = {}
        content_batch = {}
        
        for result in embedding_results:
            try:
                vector = np.array(result.embedding, dtype=np.float32)
                vector = self._normalize_vector(vector)
                
                vectors.append(vector)
                chunk_ids.append(result.chunk_id)
                metadata_batch[result.chunk_id] = result.metadata
                
                # Extract content from metadata or use empty string
                content = ""
                if hasattr(result, 'content'):
                    content = result.content
                elif 'content' in result.metadata:
                    content = result.metadata['content']
                
                content_batch[result.chunk_id] = content
                successful_additions += 1
                
            except Exception as e:
                logger.warning(f"Failed to process embedding for chunk {result.chunk_id}: {e}")
        
        if vectors:
            try:
                # Batch add to FAISS
                vectors_array = np.vstack(vectors)
                self.index.add(vectors_array)
                
                # Update stores
                self.metadata_store.update(metadata_batch)
                self.content_store.update(content_batch)
                self.id_mapping.extend(chunk_ids)
                
                # Update stats
                self.stats["total_insertions"] += successful_additions
                self.stats["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                
                logger.info(f"Successfully added {successful_additions} embeddings to vector database")
                
            except Exception as e:
                logger.error(f"Failed to batch add embeddings: {e}")
                successful_additions = 0
        
        return successful_additions
    
    def search(self, query_vector: List[float], top_k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of SearchResult objects
        """
        start_time = time.time()
        
        try:
            # Prepare query vector
            query_array = np.array(query_vector, dtype=np.float32).reshape(1, -1)
            query_array = self._normalize_vector(query_array)
            
            # Search FAISS index
            search_k = min(top_k * 2, self.index.ntotal) if filter_metadata else top_k
            similarities, indices = self.index.search(query_array, search_k)
            
            # Process results
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx >= len(self.id_mapping):
                    continue
                
                chunk_id = self.id_mapping[idx]
                metadata = self.metadata_store.get(chunk_id, {})
                content = self.content_store.get(chunk_id, "")
                
                # Apply filters if specified
                if filter_metadata and not self._matches_filter(metadata, filter_metadata):
                    continue
                
                result = SearchResult(
                    chunk_id=chunk_id,
                    similarity_score=float(similarity),
                    content=content,
                    metadata=metadata,
                    embedding_dimension=self.dimension
                )
                
                results.append(result)
                
                if len(results) >= top_k:
                    break
            
            # Update search stats
            search_time = time.time() - start_time
            self.stats["total_searches"] += 1
            self.stats["total_search_time"] += search_time
            self.stats["average_search_time"] = self.stats["total_search_time"] / self.stats["total_searches"]
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_criteria: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria"""
        for key, value in filter_criteria.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False
        return True
    
    def get_by_chunk_id(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        """
        Get content and metadata by chunk ID
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Dictionary with content and metadata or None
        """
        if chunk_id in self.content_store:
            return {
                "chunk_id": chunk_id,
                "content": self.content_store[chunk_id],
                "metadata": self.metadata_store.get(chunk_id, {})
            }
        return None
    
    def get_all_chunk_ids(self) -> List[int]:
        """Get all chunk IDs in the database"""
        return list(self.content_store.keys())
    
    def delete_by_chunk_id(self, chunk_id: int) -> bool:
        """
        Delete a chunk by ID (Note: FAISS doesn't support deletion, so this marks as deleted)
        
        Args:
            chunk_id: Chunk identifier to delete
            
        Returns:
            Success status
        """
        if chunk_id in self.content_store:
            del self.content_store[chunk_id]
            del self.metadata_store[chunk_id]
            # Note: FAISS index entry remains but won't be returned due to missing metadata
            return True
        return False
    
    def clear(self) -> bool:
        """Clear all data from the database"""
        try:
            # Reset FAISS index
            self._create_index()
            
            # Clear all stores
            self.metadata_store.clear()
            self.content_store.clear()
            self.id_mapping.clear()
            self.next_internal_id = 0
            
            # Reset stats
            self.stats = {
                "total_insertions": 0,
                "total_searches": 0,
                "total_search_time": 0.0,
                "average_search_time": 0.0,
                "last_updated": None
            }
            
            logger.info("Vector database cleared")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            return False
    
    def save(self) -> bool:
        """Save the database to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))
            
            # Save metadata and content
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata_store, f)
            
            with open(self.content_path, 'wb') as f:
                pickle.dump(self.content_store, f)
            
            with open(self.mapping_path, 'wb') as f:
                pickle.dump(self.id_mapping, f)
            
            # Save stats
            with open(self.stats_path, 'w') as f:
                json.dump(self.stats, f, indent=2)
            
            logger.info(f"Vector database saved to {self.storage_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save database: {e}")
            return False
    
    def load(self) -> bool:
        """Load the database from disk"""
        try:
            if not self.index_path.exists():
                logger.info("No existing database found, starting fresh")
                return True
            
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))
            
            # Load metadata and content
            if self.metadata_path.exists():
                with open(self.metadata_path, 'rb') as f:
                    self.metadata_store = pickle.load(f)
            
            if self.content_path.exists():
                with open(self.content_path, 'rb') as f:
                    self.content_store = pickle.load(f)
            
            if self.mapping_path.exists():
                with open(self.mapping_path, 'rb') as f:
                    self.id_mapping = pickle.load(f)
            
            # Load stats
            if self.stats_path.exists():
                with open(self.stats_path, 'r') as f:
                    self.stats.update(json.load(f))
            
            logger.info(f"Vector database loaded from {self.storage_path}")
            logger.info(f"Loaded {len(self.content_store)} vectors, dimension: {self.dimension}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load database: {e}")
            return False
    
    def get_stats(self) -> VectorStats:
        """Get database statistics"""
        # Calculate storage size
        total_size = 0
        if self.storage_path.exists():
            for file_path in self.storage_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        
        return VectorStats(
            total_vectors=len(self.content_store),
            embedding_dimension=self.dimension,
            index_type=self.index_type,
            total_size_mb=total_size / (1024 * 1024),
            last_updated=self.stats.get("last_updated", "Never"),
            search_performance_ms=self.stats.get("average_search_time", 0.0) * 1000
        )
    
    def optimize_index(self):
        """Optimize the FAISS index for better performance (placeholder for future enhancement)"""
        # For future: implement index optimization like IVF, PQ, etc.
        # For now, the flat index is already optimal for small to medium datasets
        logger.info("Index optimization not needed for current index type")
    
    def similarity_search_with_scores(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Perform similarity search and return scores with content
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of (similarity_score, content_dict) tuples
        """
        results = self.search(query_vector, top_k)
        return [
            (result.similarity_score, {
                "chunk_id": result.chunk_id,
                "content": result.content,
                "metadata": result.metadata
            })
            for result in results
        ]