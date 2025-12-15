#!/usr/bin/env python3
"""
Full Pipeline Test for Crawl2Answer
Tests both crawling and text extraction capabilities
"""

import sys
import asyncio
from typing import List
import json
import os

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from crawling.crawler import WebCrawler
    from extraction.text_extractor import TextExtractor
    from config.settings import Settings
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\n💡 To fix this, install the required dependencies:")
    print("   pip install beautifulsoup4 requests")
    print("\n   Or run the setup script:")
    print("   python setup_environment.py")
    sys.exit(1)

async def test_crawling_and_extraction():
    """Test the complete crawling + extraction pipeline"""
    
    print("🚀 Testing Crawl2Answer Pipeline")
    print("=" * 50)
    
    # Initialize settings and components
    settings = Settings()
    crawler = WebCrawler()
    extractor = TextExtractor()
    
    # Test URL - small documentation section
    test_url = "https://docs.python.org/3/tutorial/introduction.html"
    
    print(f"\n📍 Target URL: {test_url}")
    print(f"🔧 Settings: {settings.max_pages} pages, depth {settings.max_depth}")
    
    try:
        print("\n🕷️  STEP 1: Web Crawling")
        print("-" * 30)
        
        # Crawl with limited scope for testing
        crawled_pages = await crawler.crawl_site(
            base_url=test_url,
            max_pages=3,  # Small test
            max_depth=1,
            delay=1.0
        )
        
        print(f"✅ Crawled {len(crawled_pages)} pages")
        for i, page in enumerate(crawled_pages, 1):
            print(f"   {i}. {page.title[:50]}..." if len(page.title) > 50 else f"   {i}. {page.title}")
            print(f"      URL: {page.url}")
        
        print("\n🧹 STEP 2: Text Extraction & Cleaning")
        print("-" * 40)
        
        extracted_contents = []
        for page in crawled_pages:
            print(f"\n📄 Processing: {page.title}")
            
            # Extract clean text
            cleaned_content = extractor.extract_clean_text(
                html_content=page.content,
                url=page.url,
                title=page.title
            )
            extracted_contents.append(cleaned_content)
            
            # Print extraction results
            print(f"   📊 Content length: {len(cleaned_content.content)} characters")
            print(f"   📑 Paragraphs: {cleaned_content.metadata.get('paragraph_count', 0)}")
            print(f"   📝 Headings: {cleaned_content.metadata.get('heading_count', 0)}")
            print(f"   🔗 Links: {cleaned_content.metadata.get('link_count', 0)}")
            
            # Show first few lines of clean text
            lines = cleaned_content.content.split('\n')
            preview_lines = [line.strip() for line in lines if line.strip()][:3]
            print(f"   📖 Preview:")
            for line in preview_lines:
                print(f"      {line[:80]}..." if len(line) > 80 else f"      {line}")
        
        print("\n📈 PIPELINE SUMMARY")
        print("-" * 30)
        
        total_content_length = sum(len(content.content) for content in extracted_contents)
        total_words = sum(len(content.content.split()) for content in extracted_contents)
        
        print(f"✅ Pages crawled: {len(crawled_pages)}")
        print(f"✅ Pages extracted: {len(extracted_contents)}")
        print(f"✅ Total content: {total_content_length:,} characters")
        print(f"✅ Total words: {total_words:,} words")
        print(f"✅ Average per page: {total_words // len(extracted_contents) if extracted_contents else 0:,} words")
        
        # Save results for inspection
        results = {
            "crawled_pages": [
                {
                    "url": page.url,
                    "title": page.title,
                    "content_length": len(page.content),
                    "crawl_depth": page.depth
                }
                for page in crawled_pages
            ],
            "extracted_contents": [
                {
                    "url": content.url,
                    "title": content.title,
                    "content_length": len(content.content),
                    "word_count": len(content.content.split()),
                    "metadata": content.metadata
                }
                for content in extracted_contents
            ],
            "summary": {
                "pages_processed": len(crawled_pages),
                "total_content_chars": total_content_length,
                "total_words": total_words
            }
        }
        
        with open('test_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: test_results.json")
        print("\n🎉 Pipeline test completed successfully!")
        
        return extracted_contents
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None

async def test_single_extraction():
    """Test text extraction on a single page"""
    
    print("\n🧪 BONUS: Single Page Extraction Test")
    print("-" * 40)
    
    try:
        extractor = TextExtractor()
        test_url = "https://docs.python.org/3/tutorial/introduction.html"
        
        print(f"📍 Testing URL: {test_url}")
        
        # Direct extraction test
        content = await extractor.extract_from_url(test_url)
        
        if content:
            print(f"✅ Extraction successful!")
            print(f"   📊 Content: {len(content.content)} chars")
            print(f"   📝 Title: {content.title}")
            print(f"   📑 Description: {content.description[:100] if content.description else 'None'}...")
            
            # Show clean text sample
            preview = content.content[:300].replace('\n', ' ')
            print(f"   📖 Sample: {preview}...")
            
            return content
        else:
            print("❌ Extraction failed")
            return None
            
    except Exception as e:
        print(f"❌ Single extraction error: {e}")
        return None

def main():
    """Run all tests"""
    print("🚀 Crawl2Answer Full Pipeline Test")
    print("=" * 50)
    print("This test will:")
    print("  1. Crawl a small set of web pages")
    print("  2. Extract and clean text content")
    print("  3. Generate extraction statistics")
    print("  4. Save results to test_results.json")
    
    try:
        # Test the full pipeline
        asyncio.run(test_crawling_and_extraction())
        
        # Test single extraction
        asyncio.run(test_single_extraction())
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()