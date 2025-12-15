"""
Advanced Embeddings Module

This module provides comprehensive embedding generation capabilities
supporting multiple models and optimized batch processing for the
Crawl2Answer RAG system.

Features:
- Multiple embedding models (SentenceTransformers, OpenAI)
- Batch processing for efficiency
- Embedding caching and persistence
- Dimension normalization
- Error handling and recovery
"""

import numpy as np
from typing import List, Optional, Union, Dict, Any
import logging
from pathlib import Path
import pickle
import hashlib
import time
import os
from dataclasses import dataclass

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from config.settings import Settings
from chunking.chunker import TextChunk

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    embedding: List[float]
    model_name: str
    dimension: int
    chunk_id: int
    text_hash: str
    generation_time: float
    metadata: Dict[str, Any]


class Embedder:
    """
    Advanced embedding generator with multiple model support and optimization features
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the embedder
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings or Settings()
        self.model_type = self.settings.EMBEDDING_MODEL_TYPE
        self.model_name = self.settings.EMBEDDING_MODEL_NAME
        self.model = None
        self.cache_dir = Path("data/embeddings_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.embedding_stats = {
            "total_embeddings": 0,
            "cache_hits": 0,
            "total_time": 0.0,
            "average_time": 0.0
        }
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model based on configuration"""
        
        if self.model_type == "sentence_transformers":
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("sentence-transformers not available. Install with: pip install sentence-transformers")
            
            try:
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded SentenceTransformer model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer model: {e}")
                # Fall back to a smaller model if the requested one fails
                fallback_model = "all-MiniLM-L6-v2"
                if self.model_name != fallback_model:
                    logger.info(f"Trying fallback model: {fallback_model}")
                    try:
                        self.model = SentenceTransformer(fallback_model)
                        self.model_name = fallback_model
                        logger.info(f"Successfully loaded fallback model: {fallback_model}")
                    except Exception as fallback_error:
                        logger.error(f"Failed to load fallback model: {fallback_error}")
                        raise
                else:
                    raise
        
        elif self.model_type == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai not available. Install with: pip install openai")
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI embeddings")
            
            # Use newer OpenAI client if available
            try:
                self.client = openai.OpenAI(api_key=api_key)
                self.use_new_client = True
            except AttributeError:
                openai.api_key = api_key
                self.use_new_client = False
            
            logger.info(f"Initialized OpenAI embeddings with model: {self.model_name}")
        
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}. Use 'sentence_transformers' or 'openai'")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from this model"""
        
        if self.model_type == "sentence_transformers":
            return self.model.get_sentence_embedding_dimension()
        elif self.model_type == "openai":
            # Standard dimensions for OpenAI models
            model_dimensions = {
                "text-embedding-ada-002": 1536,
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072
            }
            return model_dimensions.get(self.model_name, 1536)  # Default to ada-002
        else:
            return 384  # Default fallback
    
    def _get_text_hash(self, text: str) -> str:
        """Generate a hash for caching purposes"""
        return hashlib.md5(f"{text}_{self.model_name}".encode()).hexdigest()
    
    def _load_from_cache(self, text_hash: str) -> Optional[List[float]]:
        """Load embedding from cache if available"""
        cache_file = self.cache_dir / f"{text_hash}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {e}")
        return None
    
    def _save_to_cache(self, text_hash: str, embedding: List[float]):
        """Save embedding to cache"""
        cache_file = self.cache_dir / f"{text_hash}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to save embedding to cache: {e}")
    
    def generate_embedding(self, text: str, use_cache: bool = True) -> Optional[EmbeddingResult]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            use_cache: Whether to use caching
            
        Returns:
            EmbeddingResult object or None if failed
        """
        start_time = time.time()
        text_hash = self._get_text_hash(text)
        
        # Try cache first
        if use_cache:
            cached_embedding = self._load_from_cache(text_hash)
            if cached_embedding is not None:
                self.embedding_stats["cache_hits"] += 1
                return EmbeddingResult(
                    embedding=cached_embedding,
                    model_name=self.model_name,
                    dimension=len(cached_embedding),
                    chunk_id=-1,  # Will be set later
                    text_hash=text_hash,
                    generation_time=0.0,
                    metadata={"cached": True}
                )
        
        # Generate new embedding
        try:
            if self.model_type == "sentence_transformers":
                embedding = self._generate_sentence_transformer_embedding(text)
            elif self.model_type == "openai":
                embedding = self._generate_openai_embedding(text)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            generation_time = time.time() - start_time
            
            # Update stats
            self.embedding_stats["total_embeddings"] += 1
            self.embedding_stats["total_time"] += generation_time
            self.embedding_stats["average_time"] = self.embedding_stats["total_time"] / self.embedding_stats["total_embeddings"]
            
            # Cache the result
            if use_cache:
                self._save_to_cache(text_hash, embedding)
            
            return EmbeddingResult(
                embedding=embedding,
                model_name=self.model_name,
                dimension=len(embedding),
                chunk_id=-1,  # Will be set later
                text_hash=text_hash,
                generation_time=generation_time,
                metadata={"cached": False}
            )
        
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def _generate_sentence_transformer_embedding(self, text: str) -> List[float]:
        """Generate embedding using SentenceTransformer"""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
    def _generate_openai_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API"""
        try:
            if self.use_new_client:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model_name
                )
                return response.data[0].embedding
            else:
                response = openai.Embedding.create(
                    input=text,
                    model=self.model_name
                )
                return response['data'][0]['embedding']
        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {e}")
            raise
    
    def generate_embeddings_for_chunks(self, chunks: List[Union[TextChunk, Dict[str, Any]]], batch_size: int = 32, use_cache: bool = True) -> List[EmbeddingResult]:
        """
        Generate embeddings for text chunks with batch processing
        
        Args:
            chunks: List of TextChunk objects or dictionaries
            batch_size: Batch size for processing
            use_cache: Whether to use caching
            
        Returns:
            List of EmbeddingResult objects with chunk IDs set
        """
        results = []
        
        # Process in batches for efficiency
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            
            # Extract text and prepare metadata for each chunk
            batch_data = []
            for j, chunk in enumerate(batch_chunks):
                if isinstance(chunk, dict):
                    # Handle dictionary format
                    text = chunk.get('content', '')
                    chunk_id = chunk.get('chunk_id', i + j)
                    metadata = chunk.get('metadata', {})
                    metadata['content'] = text  # Ensure content is in metadata
                else:
                    # Handle TextChunk object
                    text = chunk.content
                    chunk_id = chunk.chunk_id
                    metadata = {
                        "content": text,
                        "chunk_url": getattr(chunk, 'url', ''),
                        "chunk_title": getattr(chunk, 'title', ''),
                        "chunk_word_count": getattr(chunk, 'word_count', 0),
                        "chunk_start_pos": getattr(chunk, 'start_pos', 0),
                        "chunk_end_pos": getattr(chunk, 'end_pos', 0)
                    }
                
                batch_data.append((text, chunk_id, metadata))
            
            batch_texts = [data[0] for data in batch_data]
            
            if self.model_type == "sentence_transformers":
                # Batch processing is efficient for sentence-transformers
                batch_results = self._generate_batch_sentence_transformers(batch_texts, use_cache)
                
                # Set chunk IDs and metadata
                for result, (text, chunk_id, metadata) in zip(batch_results, batch_data):
                    result.chunk_id = chunk_id
                    result.metadata.update(metadata)
                
                results.extend(batch_results)
            else:
                # For OpenAI, process individually (rate limiting)
                for text, chunk_id, metadata in batch_data:
                    result = self.generate_embedding(text, metadata, use_cache)
                    if result:
                        result.chunk_id = chunk_id
                        results.append(result)
                    time.sleep(0.1)  # Rate limiting for OpenAI
        
        return results
    
    def _generate_batch_sentence_transformers(self, texts: List[str], use_cache: bool) -> List[EmbeddingResult]:
        """Generate batch embeddings using SentenceTransformers"""
        results = []
        
        # Check cache first
        cached_results = {}
        uncached_texts = []
        uncached_indices = []
        
        if use_cache:
            for i, text in enumerate(texts):
                text_hash = self._get_text_hash(text)
                cached_embedding = self._load_from_cache(text_hash)
                if cached_embedding is not None:
                    cached_results[i] = EmbeddingResult(
                        embedding=cached_embedding,
                        model_name=self.model_name,
                        dimension=len(cached_embedding),
                        chunk_id=-1,
                        text_hash=text_hash,
                        generation_time=0.0,
                        metadata={"cached": True}
                    )
                    self.embedding_stats["cache_hits"] += 1
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            start_time = time.time()
            embeddings = self.model.encode(uncached_texts, normalize_embeddings=True)
            generation_time = time.time() - start_time
            
            for j, (text, embedding) in enumerate(zip(uncached_texts, embeddings)):
                text_hash = self._get_text_hash(text)
                embedding_list = embedding.tolist()
                
                result = EmbeddingResult(
                    embedding=embedding_list,
                    model_name=self.model_name,
                    dimension=len(embedding_list),
                    chunk_id=-1,
                    text_hash=text_hash,
                    generation_time=generation_time / len(uncached_texts),
                    metadata={"cached": False}
                )
                
                cached_results[uncached_indices[j]] = result
                
                # Cache the result
                if use_cache:
                    self._save_to_cache(text_hash, embedding_list)
            
            # Update stats
            self.embedding_stats["total_embeddings"] += len(uncached_texts)
            self.embedding_stats["total_time"] += generation_time
            self.embedding_stats["average_time"] = self.embedding_stats["total_time"] / self.embedding_stats["total_embeddings"]
        
        # Return results in original order
        return [cached_results[i] for i in range(len(texts)) if i in cached_results]
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding generation statistics"""
        cache_hit_rate = 0.0
        if self.embedding_stats["total_embeddings"] > 0:
            cache_hit_rate = self.embedding_stats["cache_hits"] / (self.embedding_stats["total_embeddings"] + self.embedding_stats["cache_hits"]) * 100
        
        return {
            **self.embedding_stats,
            "model_type": self.model_type,
            "model_name": self.model_name,
            "embedding_dimension": self.get_embedding_dimension(),
            "cache_hit_rate": cache_hit_rate
        }
    
    def clear_cache(self):
        """Clear the embedding cache"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Embedding cache cleared")
    
    def similarity_search_preview(self, query: str, embeddings: List[List[float]], texts: List[str], top_k: int = 5) -> List[Dict]:
        """
        Preview similarity search without vector database
        
        Args:
            query: Search query
            embeddings: List of embedding vectors
            texts: Corresponding texts
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        query_result = self.generate_embedding(query)
        if not query_result:
            return []
        
        query_embedding = np.array(query_result.embedding)
        
        # Calculate similarities
        similarities = []
        for i, embedding in enumerate(embeddings):
            embedding_array = np.array(embedding)
            # Cosine similarity
            similarity = np.dot(query_embedding, embedding_array) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding_array)
            )
            similarities.append({
                "index": i,
                "similarity": float(similarity),
                "text": texts[i][:200] + "..." if len(texts[i]) > 200 else texts[i]
            })
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]