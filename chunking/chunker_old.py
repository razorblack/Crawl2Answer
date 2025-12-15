"""
Text Chunking Module

This module handles the segmentation of extracted text into manageable chunks
for efficient embedding generation and retrieval in the RAG pipeline.

Features:
- Configurable chunk sizes and overlap
- Smart boundary detection (sentences, paragraphs)
- Metadata preservation
- Multiple chunking strategies
- Quality filtering
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import re
import logging
from config.settings import Settings
from extraction.text_extractor import CleanedContent

logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    
    content: str
    url: str
    title: str
    chunk_id: int
    start_pos: int
    end_pos: int
    chunk_size: int
    metadata: Dict[str, Any]
    
    @property
    def word_count(self) -> int:
        """Count words in this chunk"""
        return len(self.content.split())
    
    @property
    def char_count(self) -> int:
        """Count characters in this chunk"""
        return len(self.content)


class TextChunker:
    """
    Advanced text chunking with multiple strategies and smart boundary detection
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the chunker
        
        Args:
            settings: Configuration settings for chunking
        """
        self.settings = settings or Settings()
        self.chunk_size = self.settings.chunk_size
        self.chunk_overlap = self.settings.chunk_overlap
        
        # Sentence boundary patterns
        self.sentence_endings = re.compile(r'[.!?]+\s+')
        self.paragraph_breaks = re.compile(r'\n\s*\n')
        
        # Quality filters
        self.min_chunk_size = 50  # Minimum meaningful chunk size
        self.max_chunk_size = self.chunk_size * 2  # Maximum before forced split
    
    def chunk_content(self, content: CleanedContent, strategy: str = "smart") -> List[TextChunk]:
        """
        Chunk the cleaned content using specified strategy
        
        Args:
            content: Cleaned content to chunk
            strategy: Chunking strategy ('smart', 'fixed', 'sentence', 'paragraph')
            
        Returns:
            List of text chunks
        """
        
        if not content or not content.content:
            return []
        
        # Select chunking strategy
        if strategy == "smart":
            return self._smart_chunk(content)
        elif strategy == "fixed":
            return self._fixed_chunk(content)
        elif strategy == "sentence":
            return self._sentence_chunk(content)
        elif strategy == "paragraph":
            return self._paragraph_chunk(content)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    def _smart_chunk(self, content: CleanedContent) -> List[TextChunk]:
        """
        Smart chunking that tries to respect sentence and paragraph boundaries
        while maintaining target chunk sizes
        """
        
        chunks = []
        text = content.content
        
        # Split by paragraphs first
        paragraphs = self.paragraph_breaks.split(text)
        
        current_chunk = ""
        current_start = 0
        chunk_id = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                
                # Create chunk from current content
                chunk = self._create_chunk(
                    content=content,
                    text=current_chunk.strip(),
                    chunk_id=chunk_id,
                    start_pos=current_start,
                    end_pos=current_start + len(current_chunk)
                )
                
                if chunk:
                    chunks.append(chunk)
                    chunk_id += 1
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                overlap_text = current_chunk[overlap_start:] if self.chunk_overlap > 0 else ""
                
                current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
                current_start = current_start + len(current_chunk) - len(current_chunk)
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk.strip():
            chunk = self._create_chunk(
                content=content,
                text=current_chunk.strip(),
                chunk_id=chunk_id,
                start_pos=current_start,
                end_pos=current_start + len(current_chunk)
            )
            
            if chunk:
                chunks.append(chunk)
        
        return self._post_process_chunks(chunks)
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