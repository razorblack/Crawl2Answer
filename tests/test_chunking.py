#!/usr/bin/env python3
"""
Text Chunking Test Script

Tests the text chunking functionality with different strategies
and analyzes the results.
"""

import sys
import json
from typing import List
import os

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from extraction.text_extractor import CleanedContent
    from chunking.chunker import TextChunker, TextChunk
    from config.settings import Settings
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\n💡 To fix this, make sure all modules are properly set up.")
    print("   This test requires the extraction and chunking modules.")
    sys.exit(1)

def create_sample_content() -> CleanedContent:
    """Create sample content for testing"""
    
    sample_text = """
    Introduction to Python Programming
    
    Python is a high-level programming language that emphasizes code readability and simplicity. It was created by Guido van Rossum and first released in 1991. Python's design philosophy emphasizes code readability with its notable use of significant whitespace.
    
    Getting Started with Python
    
    To start programming in Python, you need to install Python on your computer. You can download Python from the official website at python.org. The installation process is straightforward and well-documented.
    
    Basic Python Syntax
    
    Python syntax is designed to be readable and clean. Unlike many other programming languages, Python uses indentation to define code blocks instead of curly braces. This makes Python code more readable and forces good programming practices.
    
    Variables and Data Types
    
    In Python, you don't need to declare variables before using them. Python has several built-in data types including integers, floats, strings, lists, tuples, and dictionaries. Each data type has its own characteristics and use cases.
    
    Control Structures
    
    Python provides several control structures including if statements, for loops, while loops, and exception handling. These structures allow you to control the flow of your program and handle different conditions and scenarios.
    
    Functions and Modules
    
    Functions in Python are defined using the def keyword. They allow you to organize your code into reusable blocks. Modules are Python files that contain functions, classes, and variables. They help you organize your code and share functionality across different programs.
    
    Object-Oriented Programming
    
    Python supports object-oriented programming (OOP) with classes and objects. OOP allows you to create custom data types and organize your code in a more structured way. This is particularly useful for larger applications.
    
    Error Handling
    
    Python provides robust error handling mechanisms through try-except blocks. This allows your programs to handle unexpected situations gracefully without crashing.
    
    Conclusion
    
    Python is a versatile and powerful programming language that is excellent for beginners and experienced programmers alike. Its simple syntax and extensive library ecosystem make it suitable for a wide range of applications including web development, data science, artificial intelligence, and automation.
    """
    
    return CleanedContent(
        content=sample_text.strip(),
        url="https://example.com/python-tutorial",
        title="Introduction to Python Programming",
        description="A comprehensive guide to getting started with Python programming",
        metadata={
            "author": "Tutorial Team",
            "tags": ["python", "programming", "tutorial"],
            "difficulty": "beginner",
            "estimated_reading_time": "5 minutes"
        }
    )

def test_chunking_strategy(chunker: TextChunker, content: CleanedContent, 
                          strategy: str) -> List[TextChunk]:
    """Test a specific chunking strategy"""
    
    print(f"\n🔧 Testing {strategy.upper()} chunking strategy")
    print("-" * 40)
    
    chunks = chunker.chunk_content(content, strategy)
    
    print(f"✅ Generated {len(chunks)} chunks")
    
    # Display chunk information
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"\n📄 Chunk {i + 1}:")
        print(f"   📊 Size: {chunk.chunk_size} chars, {chunk.word_count} words")
        print(f"   🎯 Position: {chunk.start_pos}-{chunk.end_pos}")
        
        # Show preview
        preview = chunk.content[:150].replace('\n', ' ')
        print(f"   📖 Preview: {preview}...")
    
    if len(chunks) > 3:
        print(f"\n   ... and {len(chunks) - 3} more chunks")
    
    # Get statistics
    stats = chunker.get_chunking_stats(chunks)
    print(f"\n📈 Statistics:")
    print(f"   • Total chunks: {stats['total_chunks']}")
    print(f"   • Avg size: {stats['avg_chunk_size']:.0f} chars")
    print(f"   • Avg words: {stats['avg_word_count']:.0f} words")
    print(f"   • Size range: {stats['min_chunk_size']}-{stats['max_chunk_size']} chars")
    
    return chunks

def compare_strategies(chunker: TextChunker, content: CleanedContent):
    """Compare different chunking strategies"""
    
    print("\n📊 STRATEGY COMPARISON")
    print("=" * 50)
    
    strategies = ["smart", "fixed", "sentence", "paragraph"]
    results = {}
    
    for strategy in strategies:
        try:
            chunks = test_chunking_strategy(chunker, content, strategy)
            stats = chunker.get_chunking_stats(chunks)
            results[strategy] = stats
        except Exception as e:
            print(f"❌ Error testing {strategy} strategy: {e}")
            results[strategy] = {"error": str(e)}
    
    print("\n📋 SUMMARY COMPARISON")
    print("-" * 30)
    
    for strategy, stats in results.items():
        if "error" not in stats:
            print(f"{strategy.capitalize():>12}: {stats['total_chunks']:>3} chunks, "
                  f"avg {stats['avg_chunk_size']:>4.0f} chars, "
                  f"avg {stats['avg_word_count']:>3.0f} words")
        else:
            print(f"{strategy.capitalize():>12}: ❌ {stats['error']}")
    
    return results

def test_chunk_quality(chunks: List[TextChunk]):
    """Analyze chunk quality"""
    
    print("\n🔍 CHUNK QUALITY ANALYSIS")
    print("-" * 35)
    
    if not chunks:
        print("❌ No chunks to analyze")
        return
    
    # Analyze content distribution
    word_counts = [chunk.word_count for chunk in chunks]
    char_counts = [chunk.chunk_size for chunk in chunks]
    
    print(f"📊 Content Distribution:")
    print(f"   • Word count std dev: {_calculate_std(word_counts):.1f}")
    print(f"   • Char count std dev: {_calculate_std(char_counts):.1f}")
    
    # Check for empty or low-quality chunks
    low_quality = [chunk for chunk in chunks if chunk.word_count < 20]
    if low_quality:
        print(f"⚠️  Found {len(low_quality)} low-quality chunks (< 20 words)")
    else:
        print("✅ All chunks have good word count")
    
    # Check overlap quality
    overlap_info = []
    for i in range(len(chunks) - 1):
        current_end = chunks[i].content[-50:] if len(chunks[i].content) > 50 else chunks[i].content
        next_start = chunks[i + 1].content[:50] if len(chunks[i + 1].content) > 50 else chunks[i + 1].content
        
        # Simple overlap detection
        overlap = len(set(current_end.split()) & set(next_start.split()))
        overlap_info.append(overlap)
    
    if overlap_info:
        avg_overlap = sum(overlap_info) / len(overlap_info)
        print(f"🔗 Average word overlap between chunks: {avg_overlap:.1f} words")

def _calculate_std(values: List[int]) -> float:
    """Calculate standard deviation"""
    if len(values) < 2:
        return 0.0
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5

def main():
    """Run comprehensive chunking tests"""
    
    print("🚀 Text Chunking Test Suite")
    print("=" * 50)
    
    try:
        # Initialize components
        settings = Settings()
        print(f"⚙️  Settings: chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap}")
        
        chunker = TextChunker(settings)
        content = create_sample_content()
        
        print(f"\n📄 Sample content: {len(content.content)} characters, {len(content.content.split())} words")
        
        # Test different strategies
        strategy_results = compare_strategies(chunker, content)
        
        # Detailed analysis of best strategy
        print("\n🎯 DETAILED ANALYSIS (Smart Strategy)")
        print("=" * 45)
        
        smart_chunks = chunker.chunk_content(content, "smart")
        test_chunk_quality(smart_chunks)
        
        # Save results
        test_results = {
            "content_info": {
                "length": len(content.content),
                "word_count": len(content.content.split()),
                "title": content.title,
                "url": content.url
            },
            "settings": {
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap
            },
            "strategy_comparison": strategy_results,
            "detailed_chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "word_count": chunk.word_count,
                    "char_count": chunk.chunk_size,
                    "start_pos": chunk.start_pos,
                    "end_pos": chunk.end_pos,
                    "metadata": chunk.metadata
                }
                for chunk in smart_chunks
            ]
        }
        
        with open('chunking_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Test results saved to: chunking_test_results.json")
        print("\n🎉 Chunking test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()