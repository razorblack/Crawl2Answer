"""
Vector database module for storing and querying embeddings.
"""

import numpy as np
import pickle
import json
import os
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import faiss

logger = logging.getLogger(__name__)


class VectorDatabase:
    """Simple vector database using FAISS for similarity search."""
    
    def __init__(self, dimension: int, storage_path: str = "data/embeddings"):
        """
        Initialize the vector database.
        
        Args:
            dimension: Dimension of the embedding vectors
            storage_path: Path to store the database files
        """
        self.dimension = dimension
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(dimension)  # Inner product similarity
        
        # Store metadata separately
        self.metadata = []
        self.id_to_idx = {}  # Map chunk IDs to index positions
        
        # File paths
        self.index_path = self.storage_path / "faiss_index.bin"
        self.metadata_path = self.storage_path / "metadata.json"
        
    def add_embeddings(self, chunks: List[Dict]) -> bool:
        """
        Add embeddings and metadata to the database.
        
        Args:
            chunks: List of chunk dictionaries with embeddings
            
        Returns:
            True if successful, False otherwise
        """
        try:
            vectors = []
            valid_chunks = []
            
            for chunk in chunks:
                if chunk.get('embedding') is not None:
                    vectors.append(chunk['embedding'])
                    valid_chunks.append(chunk)
            
            if not vectors:
                logger.warning("No valid embeddings to add")
                return False
            
            # Convert to numpy array and normalize for cosine similarity
            vectors_np = np.array(vectors, dtype='float32')
            faiss.normalize_L2(vectors_np)
            
            # Add to FAISS index
            start_idx = self.index.ntotal
            self.index.add(vectors_np)
            
            # Store metadata
            for i, chunk in enumerate(valid_chunks):
                idx = start_idx + i
                chunk_id = chunk.get('id', len(self.metadata))
                
                self.id_to_idx[chunk_id] = idx
                
                metadata_entry = {
                    'chunk_id': chunk_id,
                    'content': chunk['content'],
                    'char_count': chunk.get('char_count', 0),
                    'word_count': chunk.get('word_count', 0),
                    'metadata': chunk.get('metadata', {}),
                    'embedding_model': chunk.get('embedding_model', 'unknown')
                }
                self.metadata.append(metadata_entry)
            
            logger.info(f"Added {len(valid_chunks)} embeddings to vector database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add embeddings: {e}")
            return False
    
    def search(self, query_embedding: List[float], k: int = 5, 
               score_threshold: float = 0.0) -> List[Dict]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query vector to search for
            k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of search results with metadata and scores
        """
        if self.index.ntotal == 0:
            logger.warning("Vector database is empty")
            return []
        
        try:
            # Normalize query vector
            query_np = np.array([query_embedding], dtype='float32')
            faiss.normalize_L2(query_np)
            
            # Search
            scores, indices = self.index.search(query_np, min(k, self.index.ntotal))
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1 or score < score_threshold:
                    continue
                
                if idx < len(self.metadata):
                    result = self.metadata[idx].copy()
                    result['similarity_score'] = float(score)
                    result['rank'] = i + 1
                    results.append(result)
            
            logger.info(f"Found {len(results)} similar embeddings")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def save(self) -> bool:
        """
        Save the database to disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))
            
            # Save metadata
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': self.metadata,
                    'id_to_idx': self.id_to_idx,
                    'dimension': self.dimension
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved vector database to {self.storage_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save database: {e}")
            return False
    
    def load(self) -> bool:
        """
        Load the database from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.index_path.exists() or not self.metadata_path.exists():
                logger.info("No existing database found, starting fresh")
                return True
            
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))
            
            # Load metadata
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.metadata = data['metadata']
                self.id_to_idx = {int(k): v for k, v in data['id_to_idx'].items()}
                
            logger.info(f"Loaded vector database with {len(self.metadata)} entries")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load database: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'metadata_entries': len(self.metadata),
            'storage_path': str(self.storage_path)
        }
    
    def clear(self) -> bool:
        """
        Clear all data from the database.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []
            self.id_to_idx = {}
            
            # Remove files if they exist
            if self.index_path.exists():
                self.index_path.unlink()
            if self.metadata_path.exists():
                self.metadata_path.unlink()
                
            logger.info("Cleared vector database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            return False