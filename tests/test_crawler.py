#!/usr/bin/env python3
"""
Test script for the enhanced web crawler.
This script demonstrates the crawling functionality by crawling a website
and printing the results.
"""

import sys
import os
import logging

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crawling.crawler import WebCrawler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_crawler(base_url: str, max_pages: int = 5, max_depth: int = 2):
    """
    Test the web crawler with a given URL.
    
    Args:
        base_url: URL to start crawling from
        max_pages: Maximum number of pages to crawl
        max_depth: Maximum crawling depth
    """
    print(f"\n🚀 Testing Web Crawler")
    print(f"Target URL: {base_url}")
    print(f"Max Pages: {max_pages}")
    print(f"Max Depth: {max_depth}")
    print(f"{'='*60}")
    
    try:
        # Initialize crawler
        crawler = WebCrawler(
            base_url=base_url,
            delay=1.0,  # 1 second delay between requests
            max_depth=max_depth
        )
        
        # Crawl the website
        crawled_pages = crawler.crawl_site(max_pages=max_pages)
        
        # Print results
        crawler.print_crawled_urls(crawled_pages)
        
        # Get and display statistics
        stats = crawler.get_crawl_stats(crawled_pages)
        print(f"\n📊 CRAWLING STATISTICS")
        print(f"{'='*60}")
        print(f"Total pages crawled: {stats['total_pages']}")
        print(f"Base domain: {stats['base_domain']}")
        print(f"Total content size: {stats['total_content_size']:,} characters")
        print(f"Average content size: {stats['avg_content_size']:,.0f} characters")
        print(f"Max depth used: {stats['crawl_settings']['max_depth']}")
        print(f"Request delay: {stats['crawl_settings']['delay']} seconds")
        
        return crawled_pages
        
    except Exception as e:
        print(f"❌ Error during crawling: {e}")
        return []

def main():
    """Main function to run crawler tests."""
    
    # Test websites (choose one or modify as needed)
    test_sites = [
        {
            "name": "Python Documentation",
            "url": "https://docs.python.org/3/tutorial/",
            "max_pages": 8,
            "max_depth": 2
        },
        {
            "name": "Wikipedia - Python Programming",
            "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
            "max_pages": 5,
            "max_depth": 1
        },
        {
            "name": "Real Python",
            "url": "https://realpython.com/python-basics/",
            "max_pages": 6,
            "max_depth": 2
        }
    ]
    
    # You can change this index to test different sites
    site_index = 0  # Change to 1 or 2 to test other sites
    
    if site_index < len(test_sites):
        site = test_sites[site_index]
        print(f"🌐 Testing with: {site['name']}")
        
        crawled_pages = test_crawler(
            base_url=site['url'],
            max_pages=site['max_pages'],
            max_depth=site['max_depth']
        )
        
        if crawled_pages:
            print(f"\n✅ Crawling test completed successfully!")
            print(f"Found {len(crawled_pages)} pages with content.")
        else:
            print(f"\n❌ Crawling test failed or found no pages.")
    else:
        print("❌ Invalid site index. Please check the site_index value.")

if __name__ == "__main__":
    main()