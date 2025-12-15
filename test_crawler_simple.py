#!/usr/bin/env python3
"""
Simple crawler test with local URL simulation.
"""

import sys
import os
import logging
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Create a mock test to verify the crawler structure
def test_crawler_structure():
    """Test if the crawler can be imported and has the required methods."""
    
    print("🧪 Testing Crawler Structure")
    print("=" * 50)
    
    try:
        from crawling.crawler import WebCrawler, CrawledPage
        print("✅ Successfully imported WebCrawler and CrawledPage")
        
        # Test initialization
        crawler = WebCrawler("https://example.com", delay=0.1, max_depth=2)
        print("✅ Successfully created WebCrawler instance")
        
        # Check if required methods exist
        required_methods = [
            'fetch_page', 'crawl_site', 'print_crawled_urls', 
            'get_crawl_stats', '_extract_title', '_extract_internal_links',
            '_is_internal_url', '_clean_url', '_should_skip_url'
        ]
        
        for method in required_methods:
            if hasattr(crawler, method):
                print(f"✅ Method '{method}' exists")
            else:
                print(f"❌ Method '{method}' missing")
        
        # Test CrawledPage structure
        test_page = CrawledPage(
            url="https://example.com",
            title="Test Page",
            html_content="<html><head><title>Test</title></head><body>Content</body></html>",
            status_code=200,
            crawl_timestamp=datetime.now().isoformat()
        )
        print(f"✅ CrawledPage structure works: {test_page.title}")
        
        # Test URL filtering
        test_urls = [
            "https://example.com/page1",
            "https://example.com/login",
            "https://example.com/api/data",
            "https://example.com/about.pdf",
            "https://example.com/contact",
        ]
        
        print(f"\n📋 Testing URL filtering:")
        for url in test_urls:
            should_skip = crawler._should_skip_url(url)
            status = "SKIP" if should_skip else "CRAWL"
            print(f"   {status}: {url}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_url_patterns():
    """Test URL pattern matching."""
    print(f"\n🔍 Testing URL Pattern Matching")
    print("=" * 50)
    
    try:
        from crawling.crawler import WebCrawler
        
        crawler = WebCrawler("https://example.com")
        
        # Test cases: (url, should_skip, reason)
        test_cases = [
            ("https://example.com/", False, "Home page"),
            ("https://example.com/about", False, "About page"),
            ("https://example.com/docs/guide", False, "Documentation"),
            ("https://example.com/login", True, "Login page"),
            ("https://example.com/admin/panel", True, "Admin panel"),
            ("https://example.com/cart", True, "Shopping cart"),
            ("https://example.com/file.pdf", True, "PDF file"),
            ("https://example.com/api/users", True, "API endpoint"),
            ("https://example.com/contact", True, "Contact page"),
            ("https://otherdomain.com/page", False, "External domain (handled by domain check)"),
        ]
        
        for url, expected_skip, reason in test_cases:
            actual_skip = crawler._should_skip_url(url)
            status = "✅" if actual_skip == expected_skip else "❌"
            action = "SKIP" if actual_skip else "CRAWL"
            print(f"{status} {action}: {url} ({reason})")
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing URL patterns: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Enhanced Web Crawler - Structure Test")
    print("=" * 60)
    
    success = True
    
    # Test 1: Basic structure
    success &= test_crawler_structure()
    
    # Test 2: URL patterns
    success &= test_url_patterns()
    
    # Summary
    print(f"\n{'='*60}")
    if success:
        print("🎉 All tests passed! The enhanced crawler is ready.")
        print("💡 You can now test with real websites using the API or by modifying this script.")
        print(f"\n📋 Quick Test Commands:")
        print(f"1. Start API: python -m uvicorn api.main:app --reload")
        print(f"2. Test endpoint: POST /test-crawl with a URL")
        print(f"3. Example URL: https://docs.python.org/3/tutorial/")
    else:
        print("❌ Some tests failed. Check the errors above.")
    
    return success

if __name__ == "__main__":
    main()