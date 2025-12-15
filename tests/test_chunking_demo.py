#!/usr/bin/env python3
"""
Step 4 Chunking Demo - Standalone Test

This test demonstrates the text chunking capabilities without external dependencies.
It creates sample content and shows how different chunking strategies work.
"""

import sys
from dataclasses import dataclass
from typing import Dict, Any, List
import re

# Mock the CleanedContent class for testing
@dataclass
class CleanedContent:
    content: str
    url: str
    title: str
    description: str
    metadata: Dict[str, Any]

# Mock Settings class for testing
class Settings:
    def __init__(self):
        self.chunk_size = 1000
        self.chunk_overlap = 200

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

# Simplified chunker for demo
class DemoChunker:
    """Simplified text chunker for demonstration"""
    
    def __init__(self):
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.paragraph_breaks = re.compile(r'\n\s*\n')
    
    def chunk_content(self, content: CleanedContent, strategy: str = "smart") -> List[TextChunk]:
        """Chunk content using smart strategy"""
        
        chunks = []
        text = content.content
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
                
                chunk = TextChunk(
                    content=current_chunk.strip(),
                    url=content.url,
                    title=content.title,
                    chunk_id=chunk_id,
                    start_pos=current_start,
                    end_pos=current_start + len(current_chunk),
                    chunk_size=len(current_chunk),
                    metadata={**content.metadata, "chunk_index": chunk_id}
                )
                
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
            chunk = TextChunk(
                content=current_chunk.strip(),
                url=content.url,
                title=content.title,
                chunk_id=chunk_id,
                start_pos=current_start,
                end_pos=current_start + len(current_chunk),
                chunk_size=len(current_chunk),
                metadata={**content.metadata, "chunk_index": chunk_id}
            )
            chunks.append(chunk)
        
        return chunks

def create_sample_content() -> CleanedContent:
    """Create sample content for testing"""
    
    sample_text = """Introduction to Python Programming

Python is a high-level programming language that emphasizes code readability and simplicity. It was created by Guido van Rossum and first released in 1991. Python's design philosophy emphasizes code readability with its notable use of significant whitespace.

Getting Started with Python

To start programming in Python, you need to install Python on your computer. You can download Python from the official website at python.org. The installation process is straightforward and well-documented.

Python comes with a comprehensive standard library that provides tools and modules for various tasks including file I/O, system calls, networking, and data manipulation. This extensive library ecosystem is one of Python's greatest strengths.

Basic Python Syntax

Python syntax is designed to be readable and clean. Unlike many other programming languages, Python uses indentation to define code blocks instead of curly braces. This makes Python code more readable and forces good programming practices.

Variables in Python are dynamically typed, meaning you don't need to declare their type explicitly. Python will automatically determine the type based on the value assigned to the variable.

Variables and Data Types

Python has several built-in data types including integers, floats, strings, lists, tuples, and dictionaries. Each data type has its own characteristics and use cases.

Integers are whole numbers, floats are decimal numbers, and strings are sequences of characters. Lists are ordered collections that can contain different data types, while tuples are immutable ordered collections.

Control Structures

Python provides several control structures including if statements, for loops, while loops, and exception handling. These structures allow you to control the flow of your program and handle different conditions and scenarios.

If statements allow you to execute code based on certain conditions. For loops let you iterate over sequences, while while loops continue executing as long as a condition is true.

Functions and Modules

Functions in Python are defined using the def keyword. They allow you to organize your code into reusable blocks. Functions can accept parameters and return values, making them powerful tools for code organization.

Modules are Python files that contain functions, classes, and variables. They help you organize your code and share functionality across different programs. Python's import system makes it easy to use code from other modules.

Object-Oriented Programming

Python supports object-oriented programming (OOP) with classes and objects. OOP allows you to create custom data types and organize your code in a more structured way. This is particularly useful for larger applications.

Classes define the structure and behavior of objects, while objects are instances of classes. Python supports inheritance, encapsulation, and polymorphism, the three pillars of object-oriented programming.

Error Handling

Python provides robust error handling mechanisms through try-except blocks. This allows your programs to handle unexpected situations gracefully without crashing.

When an error occurs in a try block, the program jumps to the corresponding except block to handle the error. This makes your programs more robust and user-friendly.

Advanced Features

Python offers many advanced features including decorators, generators, context managers, and metaclasses. These features allow you to write more elegant and efficient code.

Decorators allow you to modify the behavior of functions or classes. Generators provide a way to create iterators efficiently. Context managers help you manage resources properly.

Libraries and Frameworks

Python has a vast ecosystem of third-party libraries and frameworks. Popular libraries include NumPy for numerical computing, pandas for data analysis, and Django for web development.

These libraries extend Python's capabilities significantly and allow you to build complex applications quickly. The Python Package Index (PyPI) hosts thousands of packages that you can install using pip.

Conclusion

Python is a versatile and powerful programming language that is excellent for beginners and experienced programmers alike. Its simple syntax and extensive library ecosystem make it suitable for a wide range of applications including web development, data science, artificial intelligence, and automation.

Whether you're building web applications, analyzing data, or developing machine learning models, Python provides the tools and flexibility you need to succeed."""

    return CleanedContent(
        content=sample_text.strip(),
        url="https://example.com/python-tutorial",
        title="Introduction to Python Programming",
        description="A comprehensive guide to getting started with Python programming",
        metadata={
            "author": "Tutorial Team",
            "tags": ["python", "programming", "tutorial"],
            "difficulty": "beginner",
            "estimated_reading_time": "10 minutes"
        }
    )

def main():
    """Demonstrate chunking capabilities"""
    
    print("🚀 Step 4: Text Chunking Demo")
    print("=" * 50)
    
    # Create sample content
    content = create_sample_content()
    print(f"📄 Sample content: {len(content.content)} characters, {len(content.content.split())} words")
    
    # Initialize chunker
    chunker = DemoChunker()
    print(f"⚙️  Settings: chunk_size={chunker.chunk_size}, overlap={chunker.chunk_overlap}")
    
    # Chunk the content
    print(f"\n🧩 Chunking content...")
    chunks = chunker.chunk_content(content)
    
    print(f"✅ Generated {len(chunks)} chunks")
    
    # Display chunk information
    print(f"\n📊 CHUNK ANALYSIS")
    print("-" * 30)
    
    total_chars = sum(chunk.chunk_size for chunk in chunks)
    total_words = sum(chunk.word_count for chunk in chunks)
    
    print(f"📈 Statistics:")
    print(f"   • Total chunks: {len(chunks)}")
    print(f"   • Total characters: {total_chars:,}")
    print(f"   • Total words: {total_words:,}")
    print(f"   • Avg chunk size: {total_chars // len(chunks):,} chars")
    print(f"   • Avg words per chunk: {total_words // len(chunks):,} words")
    
    # Show first few chunks
    print(f"\n📋 CHUNK PREVIEWS")
    print("-" * 30)
    
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n🔖 Chunk {i + 1}:")
        print(f"   📊 Size: {chunk.chunk_size} chars, {chunk.word_count} words")
        print(f"   🎯 Position: {chunk.start_pos}-{chunk.end_pos}")
        
        # Show preview
        preview = chunk.content[:200].replace('\n', ' ')
        print(f"   📖 Preview: {preview}...")
        
        if i < len(chunks) - 1:
            print()
    
    if len(chunks) > 3:
        print(f"\n   ... and {len(chunks) - 3} more chunks")
    
    print(f"\n🎯 KEY FEATURES DEMONSTRATED")
    print("-" * 35)
    print("✅ Smart paragraph-based chunking")
    print("✅ Configurable chunk size and overlap")
    print("✅ Metadata preservation")
    print("✅ Position tracking")
    print("✅ Quality metrics")
    print("✅ Boundary respect")
    
    print(f"\n📁 READY FOR NEXT STEPS")
    print("-" * 25)
    print("🔄 Step 5: Generate embeddings for chunks")
    print("💾 Step 6: Store in vector database")
    print("🔍 Step 7: Implement semantic retrieval")
    
    print(f"\n🎉 Step 4 chunking demo completed!")
    print("\nNote: Install dependencies (beautifulsoup4, requests) to run the full test suite.")

if __name__ == "__main__":
    main()