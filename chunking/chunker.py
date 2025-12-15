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
    
    def _fixed_chunk(self, content: CleanedContent) -> List[TextChunk]:
        """
        Simple fixed-size chunking with overlap
        """
        
        chunks = []
        text = content.content
        chunk_id = 0
        
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk_text = text[i:i + self.chunk_size]
            
            chunk = self._create_chunk(
                content=content,
                text=chunk_text,
                chunk_id=chunk_id,
                start_pos=i,
                end_pos=i + len(chunk_text)
            )
            
            if chunk:
                chunks.append(chunk)
                chunk_id += 1
        
        return chunks
    
    def _sentence_chunk(self, content: CleanedContent) -> List[TextChunk]:
        """
        Chunk by sentences, respecting sentence boundaries
        """
        
        chunks = []
        sentences = self.sentence_endings.split(content.content)
        
        current_chunk = ""
        current_start = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                
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
                
                # Start new chunk with potential overlap
                current_chunk = sentence
                current_start = current_start + len(current_chunk) - len(current_chunk)
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
        
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
        
        return chunks
    
    def _paragraph_chunk(self, content: CleanedContent) -> List[TextChunk]:
        """
        Chunk by paragraphs, keeping paragraphs intact when possible
        """
        
        chunks = []
        paragraphs = self.paragraph_breaks.split(content.content)
        chunk_id = 0
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If paragraph is too large, split it further
            if len(paragraph) > self.chunk_size:
                # Split large paragraph using smart chunking
                temp_content = CleanedContent(
                    content=paragraph,
                    url=content.url,
                    title=content.title,
                    description=content.description,
                    metadata=content.metadata
                )
                
                sub_chunks = self._smart_chunk(temp_content)
                for sub_chunk in sub_chunks:
                    sub_chunk.chunk_id = chunk_id
                    chunks.append(sub_chunk)
                    chunk_id += 1
            else:
                # Use paragraph as chunk
                chunk = self._create_chunk(
                    content=content,
                    text=paragraph,
                    chunk_id=chunk_id,
                    start_pos=0,  # Simplified for paragraph mode
                    end_pos=len(paragraph)
                )
                
                if chunk:
                    chunks.append(chunk)
                    chunk_id += 1
        
        return chunks
    
    def _create_chunk(self, content: CleanedContent, text: str, chunk_id: int, 
                     start_pos: int, end_pos: int) -> Optional[TextChunk]:
        """
        Create a TextChunk object with metadata
        
        Args:
            content: Original cleaned content
            text: Chunk text
            chunk_id: Unique chunk identifier
            start_pos: Starting position in original text
            end_pos: Ending position in original text
            
        Returns:
            TextChunk object or None if invalid
        """
        
        # Quality filters
        if len(text.strip()) < self.min_chunk_size:
            return None
        
        # Clean the chunk text
        text = self._clean_chunk_text(text)
        
        if not text.strip():
            return None
        
        # Extract chunk-specific metadata
        chunk_metadata = {
            "original_length": len(content.content),
            "chunk_index": chunk_id,
            "overlap_size": self.chunk_overlap,
            "chunking_strategy": "smart",
            "word_density": len(text.split()) / len(text) if text else 0,
            **content.metadata  # Include original metadata
        }
        
        return TextChunk(
            content=text,
            url=content.url,
            title=content.title,
            chunk_id=chunk_id,
            start_pos=start_pos,
            end_pos=end_pos,
            chunk_size=len(text),
            metadata=chunk_metadata
        )
    
    def _clean_chunk_text(self, text: str) -> str:
        """
        Clean and normalize chunk text
        
        Args:
            text: Raw chunk text
            
        Returns:
            Cleaned chunk text
        """
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove isolated special characters
        text = re.sub(r'\s+[^\w\s]\s+', ' ', text)
        
        # Ensure proper sentence endings
        text = re.sub(r'([.!?])\s*$', r'\1', text)
        
        return text
    
    def _post_process_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        Post-process chunks for quality and consistency
        
        Args:
            chunks: List of raw chunks
            
        Returns:
            Processed and filtered chunks
        """
        
        processed_chunks = []
        
        for chunk in chunks:
            # Skip very small chunks
            if chunk.word_count < 10:
                continue
            
            # Skip chunks that are mostly special characters
            word_ratio = len(re.findall(r'\w+', chunk.content)) / len(chunk.content.split())
            if word_ratio < 0.5:
                continue
            
            processed_chunks.append(chunk)
        
        # Update chunk IDs to be sequential
        for i, chunk in enumerate(processed_chunks):
            chunk.chunk_id = i
        
        return processed_chunks
    
    def get_chunking_stats(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """
        Generate statistics about the chunking results
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Dictionary of statistics
        """
        
        if not chunks:
            return {"error": "No chunks provided"}
        
        chunk_sizes = [chunk.chunk_size for chunk in chunks]
        word_counts = [chunk.word_count for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "total_characters": sum(chunk_sizes),
            "total_words": sum(word_counts),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "avg_word_count": sum(word_counts) / len(word_counts),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "min_word_count": min(word_counts),
            "max_word_count": max(word_counts),
            "chunk_size_std": self._calculate_std(chunk_sizes),
            "word_count_std": self._calculate_std(word_counts)
        }
    
    def _calculate_std(self, values: List[int]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def chunk_multiple_contents(self, contents: List[CleanedContent], 
                              strategy: str = "smart") -> List[TextChunk]:
        """
        Chunk multiple cleaned contents
        
        Args:
            contents: List of cleaned contents
            strategy: Chunking strategy
            
        Returns:
            List of all chunks from all contents
        """
        
        all_chunks = []
        global_chunk_id = 0
        
        for content in contents:
            chunks = self.chunk_content(content, strategy)
            
            # Update chunk IDs to be globally unique
            for chunk in chunks:
                chunk.chunk_id = global_chunk_id
                all_chunks.append(chunk)
                global_chunk_id += 1
        
        return all_chunks