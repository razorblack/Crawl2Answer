"""
Embeddings module for generating vector representations of text.
"""

import openai
import numpy as np
from typing import List, Optional, Union
import logging
from sentence_transformers import SentenceTransformer
import os

logger = logging.getLogger(__name__)


class Embedder:
    """Generate embeddings for text chunks."""
    
    def __init__(self, model_type: str = "sentence_transformers", model_name: Optional[str] = None):
        """
        Initialize the embedder.
        
        Args:
            model_type: Type of embedding model ("openai" or "sentence_transformers")
            model_name: Specific model name to use
        """
        self.model_type = model_type
        self.model = None
        
        if model_type == "openai":
            self.model_name = model_name or "text-embedding-ada-002"
            openai.api_key = os.getenv("OPENAI_API_KEY")
            if not openai.api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI embeddings")
        
        elif model_type == "sentence_transformers":
            self.model_name = model_name or "all-MiniLM-L6-v2"
            try:
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded SentenceTransformer model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer model: {e}")
                raise
        
        else:
            raise ValueError("model_type must be 'openai' or 'sentence_transformers'")
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats, or None if failed
        """
        if not text or not text.strip():
            return None
        
        try:
            if self.model_type == "openai":
                return self._generate_openai_embedding(text)
            elif self.model_type == "sentence_transformers":
                return self._generate_st_embedding(text)
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def generate_batch_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        if self.model_type == "sentence_transformers" and len(texts) > 1:
            # Use batch processing for sentence transformers
            try:
                batch_embeddings = self.model.encode(texts, convert_to_numpy=True)
                embeddings = [emb.tolist() for emb in batch_embeddings]
            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                # Fall back to individual processing
                embeddings = [self.generate_embedding(text) for text in texts]
        else:
            # Process individually
            embeddings = [self.generate_embedding(text) for text in texts]
        
        return embeddings
    
    def _generate_openai_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API."""
        response = openai.Embedding.create(
            input=text,
            model=self.model_name
        )
        return response['data'][0]['embedding']
    
    def _generate_st_embedding(self, text: str) -> List[float]:
        """Generate embedding using SentenceTransformers."""
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        if self.model_type == "openai":
            # Ada-002 produces 1536-dimensional embeddings
            return 1536
        elif self.model_type == "sentence_transformers":
            # Get dimension from model
            return self.model.get_sentence_embedding_dimension()
        
        return 0
    
    def embed_chunks(self, chunks: List[dict]) -> List[dict]:
        """
        Generate embeddings for text chunks and add them to chunk data.
        
        Args:
            chunks: List of chunk dictionaries with 'content' key
            
        Returns:
            Updated chunks with 'embedding' field added
        """
        texts = [chunk['content'] for chunk in chunks]
        embeddings = self.generate_batch_embeddings(texts)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i]
            chunk['embedding_model'] = f"{self.model_type}:{self.model_name}"
        
        successful_embeddings = sum(1 for emb in embeddings if emb is not None)
        logger.info(f"Generated embeddings for {successful_embeddings}/{len(chunks)} chunks")
        
        return chunks