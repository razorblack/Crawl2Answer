"""
Text chunking module for breaking down large text into manageable segments.
"""

import re
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class TextChunker:
    """Chunk text into smaller, manageable pieces for embedding."""
    
    def __init__(self, chunk_size: int = 1000, overlap_size: int = 200):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            overlap_size: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
    
    def chunk_text(self, text: str, metadata: Optional[dict] = None) -> List[dict]:
        """
        Chunk text into smaller segments.
        
        Args:
            text: Text content to chunk
            metadata: Optional metadata to include with each chunk
            
        Returns:
            List of dictionaries containing chunk data
        """
        if not text or not text.strip():
            return []
        
        chunks = []
        
        # First try to split by paragraphs
        paragraphs = self._split_by_paragraphs(text)
        
        current_chunk = ""
        chunk_id = 0
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_data = self._create_chunk_data(
                    current_chunk.strip(), chunk_id, metadata
                )
                chunks.append(chunk_data)
                chunk_id += 1
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + paragraph
            else:
                # Add paragraph to current chunk
                current_chunk += (" " if current_chunk else "") + paragraph
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunk_data = self._create_chunk_data(
                current_chunk.strip(), chunk_id, metadata
            )
            chunks.append(chunk_data)
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.
        
        Args:
            text: Input text
            
        Returns:
            List of paragraph strings
        """
        # Split by double newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Filter out empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs
    
    def _get_overlap_text(self, text: str) -> str:
        """
        Get the last part of text for overlap with next chunk.
        
        Args:
            text: Current chunk text
            
        Returns:
            Overlap text
        """
        if len(text) <= self.overlap_size:
            return text
        
        # Try to find a sentence boundary within the overlap region
        overlap_start = len(text) - self.overlap_size
        sentence_pattern = r'[.!?]\s+'
        
        # Look for sentence boundaries in the overlap region
        sentences = re.split(sentence_pattern, text[overlap_start:])
        
        if len(sentences) > 1:
            # Use complete sentences for overlap
            return sentences[-1]
        else:
            # Fall back to character-based overlap
            return text[-self.overlap_size:]
    
    def _create_chunk_data(self, content: str, chunk_id: int, metadata: Optional[dict]) -> dict:
        """
        Create a chunk data dictionary.
        
        Args:
            content: Chunk text content
            chunk_id: Unique chunk identifier
            metadata: Optional metadata
            
        Returns:
            Chunk data dictionary
        """
        chunk_data = {
            'id': chunk_id,
            'content': content,
            'char_count': len(content),
            'word_count': len(content.split())
        }
        
        if metadata:
            chunk_data['metadata'] = metadata.copy()
        
        return chunk_data
    
    def chunk_by_sentences(self, text: str, max_sentences: int = 10) -> List[str]:
        """
        Alternative chunking method that splits by sentences.
        
        Args:
            text: Input text
            max_sentences: Maximum number of sentences per chunk
            
        Returns:
            List of text chunks
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        
        current_chunk = []
        for sentence in sentences:
            current_chunk.append(sentence)
            
            if len(current_chunk) >= max_sentences:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        
        # Add remaining sentences
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks