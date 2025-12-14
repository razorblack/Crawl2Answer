"""
Web crawler module for fetching web pages.
"""

import requests
from typing import Optional, List
from urllib.parse import urljoin, urlparse
import time
import logging

logger = logging.getLogger(__name__)


class WebCrawler:
    """A simple web crawler for extracting content from websites."""
    
    def __init__(self, base_url: str, delay: float = 1.0):
        """
        Initialize the web crawler.
        
        Args:
            base_url: The base URL to start crawling from
            delay: Delay between requests in seconds
        """
        self.base_url = base_url
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Crawl2Answer-Bot/1.0 (Educational Project)'
        })
    
    def fetch_page(self, url: str) -> Optional[str]:
        """
        Fetch a single web page.
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content of the page or None if failed
        """
        try:
            time.sleep(self.delay)
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None
    
    def crawl_site(self, max_pages: int = 10) -> List[tuple]:
        """
        Crawl the website starting from base_url.
        
        Args:
            max_pages: Maximum number of pages to crawl
            
        Returns:
            List of tuples (url, html_content)
        """
        crawled_pages = []
        urls_to_visit = [self.base_url]
        visited_urls = set()
        
        while urls_to_visit and len(crawled_pages) < max_pages:
            current_url = urls_to_visit.pop(0)
            
            if current_url in visited_urls:
                continue
                
            visited_urls.add(current_url)
            html_content = self.fetch_page(current_url)
            
            if html_content:
                crawled_pages.append((current_url, html_content))
                logger.info(f"Successfully crawled: {current_url}")
            
        return crawled_pages