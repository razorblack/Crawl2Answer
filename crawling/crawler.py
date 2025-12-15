"""
Web crawler module for fetching web pages.
"""

import requests
from typing import Optional, List, Set, Dict
from urllib.parse import urljoin, urlparse, urlunparse
import time
import logging
import re
from bs4 import BeautifulSoup
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CrawledPage:
    """Data structure for storing crawled page information."""
    url: str
    title: str
    html_content: str
    status_code: int
    crawl_timestamp: str


class WebCrawler:
    """Enhanced web crawler for extracting content from websites with domain restrictions and smart filtering."""
    
    def __init__(self, base_url: str, delay: float = 1.0, max_depth: int = 3):
        """
        Initialize the web crawler.
        
        Args:
            base_url: The base URL to start crawling from
            delay: Delay between requests in seconds
            max_depth: Maximum crawling depth
        """
        self.base_url = base_url
        self.delay = delay
        self.max_depth = max_depth
        
        # Parse base domain for filtering
        parsed_url = urlparse(base_url)
        self.base_domain = parsed_url.netloc
        self.base_scheme = parsed_url.scheme
        
        # Initialize session with headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Crawl2Answer-Bot/1.0 (Educational Project)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Patterns for URLs to skip
        self.skip_patterns = [
            # Authentication and user actions
            r'/login', r'/signin', r'/signup', r'/register', r'/logout',
            r'/account', r'/profile', r'/dashboard', r'/admin',
            
            # E-commerce
            r'/cart', r'/checkout', r'/payment', r'/order', r'/purchase',
            r'/shop', r'/store', r'/buy', r'/product',
            
            # File downloads and media
            r'\.(pdf|doc|docx|xls|xlsx|ppt|pptx|zip|rar|exe|dmg)$',
            r'\.(jpg|jpeg|png|gif|svg|ico|webp)$',
            r'\.(mp3|mp4|avi|mov|wmv|flv)$',
            
            # Non-content pages
            r'/api/', r'/feed', r'/rss', r'/sitemap',
            r'/privacy', r'/terms', r'/cookie',
            r'/contact', r'/about-us', r'/legal',
            
            # Dynamic and filtered content
            r'\?', r'#', r'/search', r'/filter', r'/sort',
            r'/page/', r'/p/', r'/category/', r'/tag/',
            
            # Social and external
            r'/share', r'/print', r'/email', r'/subscribe'
        ]
        
        # Compiled regex patterns for better performance
        self.skip_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.skip_patterns]
    
    def fetch_page(self, url: str) -> Optional[CrawledPage]:
        """
        Fetch a single web page and extract basic information.
        
        Args:
            url: URL to fetch
            
        Returns:
            CrawledPage object with page data or None if failed
        """
        try:
            time.sleep(self.delay)
            logger.info(f"Fetching: {url}")
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Extract title from HTML
            title = self._extract_title(response.text)
            
            # Get current timestamp
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            
            crawled_page = CrawledPage(
                url=url,
                title=title,
                html_content=response.text,
                status_code=response.status_code,
                crawl_timestamp=timestamp
            )
            
            logger.info(f"Successfully fetched: {url} (Title: {title[:50]}...)")
            return crawled_page
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")
            return None
    
    def crawl_site(self, max_pages: int = 10) -> List[CrawledPage]:
        """
        Crawl the website starting from base_url with smart link discovery.
        
        Args:
            max_pages: Maximum number of pages to crawl
            
        Returns:
            List of CrawledPage objects
        """
        crawled_pages = []
        urls_to_visit = [(self.base_url, 0)]  # (url, depth)
        visited_urls: Set[str] = set()
        
        logger.info(f"Starting crawl of {self.base_url} (max_pages: {max_pages}, max_depth: {self.max_depth})")
        
        while urls_to_visit and len(crawled_pages) < max_pages:
            current_url, depth = urls_to_visit.pop(0)
            
            # Skip if already visited or too deep
            if current_url in visited_urls or depth > self.max_depth:
                continue
            
            # Skip URLs that match skip patterns
            if self._should_skip_url(current_url):
                logger.debug(f"Skipping URL (matches skip pattern): {current_url}")
                continue
                
            visited_urls.add(current_url)
            crawled_page = self.fetch_page(current_url)
            
            if crawled_page:
                crawled_pages.append(crawled_page)
                
                # Extract links for further crawling (if not at max depth)
                if depth < self.max_depth:
                    new_urls = self._extract_internal_links(crawled_page.html_content, current_url)
                    for new_url in new_urls:
                        if new_url not in visited_urls and (new_url, depth + 1) not in urls_to_visit:
                            urls_to_visit.append((new_url, depth + 1))
        
        logger.info(f"Crawling completed. Found {len(crawled_pages)} pages")
        return crawled_pages
    
    def _extract_title(self, html_content: str) -> str:
        """
        Extract title from HTML content.
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            Page title or default if not found
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            title_tag = soup.find('title')
            if title_tag:
                return title_tag.get_text().strip()
            
            # Fallback to h1 tag
            h1_tag = soup.find('h1')
            if h1_tag:
                return h1_tag.get_text().strip()
                
            return "Untitled Page"
            
        except Exception as e:
            logger.warning(f"Failed to extract title: {e}")
            return "Untitled Page"
    
    def _extract_internal_links(self, html_content: str, base_url: str) -> List[str]:
        """
        Extract internal links from HTML content.
        
        Args:
            html_content: HTML content to parse
            base_url: Current page URL for resolving relative links
            
        Returns:
            List of internal URLs
        """
        internal_links = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find all anchor tags with href attributes
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Skip empty links, mailto, tel, etc.
                if not href or href.startswith(('mailto:', 'tel:', 'javascript:', '#')):
                    continue
                
                # Resolve relative URLs
                full_url = urljoin(base_url, href)
                
                # Check if it's an internal link
                if self._is_internal_url(full_url):
                    # Clean the URL (remove fragments, normalize)
                    clean_url = self._clean_url(full_url)
                    if clean_url and not self._should_skip_url(clean_url):
                        internal_links.append(clean_url)
            
        except Exception as e:
            logger.warning(f"Failed to extract links from {base_url}: {e}")
        
        # Remove duplicates and return
        return list(set(internal_links))
    
    def _is_internal_url(self, url: str) -> bool:
        """
        Check if a URL belongs to the same domain.
        
        Args:
            url: URL to check
            
        Returns:
            True if internal, False otherwise
        """
        try:
            parsed = urlparse(url)
            return parsed.netloc == self.base_domain or parsed.netloc == ''
        except:
            return False
    
    def _clean_url(self, url: str) -> str:
        """
        Clean and normalize a URL.
        
        Args:
            url: Raw URL
            
        Returns:
            Cleaned URL
        """
        try:
            parsed = urlparse(url)
            # Remove fragment and normalize
            cleaned = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                ''  # Remove fragment
            ))
            return cleaned
        except:
            return url
    
    def _should_skip_url(self, url: str) -> bool:
        """
        Check if a URL should be skipped based on patterns.
        
        Args:
            url: URL to check
            
        Returns:
            True if should skip, False otherwise
        """
        for regex in self.skip_regex:
            if regex.search(url):
                return True
        return False
    
    def print_crawled_urls(self, crawled_pages: List[CrawledPage]) -> None:
        """
        Print a formatted list of crawled URLs for testing.
        
        Args:
            crawled_pages: List of crawled pages
        """
        print(f"\n{'='*80}")
        print(f"CRAWLED PAGES SUMMARY")
        print(f"{'='*80}")
        print(f"Total pages crawled: {len(crawled_pages)}")
        print(f"Base domain: {self.base_domain}")
        print(f"{'='*80}")
        
        for i, page in enumerate(crawled_pages, 1):
            print(f"{i:2d}. {page.url}")
            print(f"    Title: {page.title}")
            print(f"    Status: {page.status_code}")
            print(f"    Crawled: {page.crawl_timestamp}")
            print(f"    Content size: {len(page.html_content)} characters")
            print("-" * 80)
    
    def get_crawl_stats(self, crawled_pages: List[CrawledPage]) -> Dict:
        """
        Get statistics about the crawling session.
        
        Args:
            crawled_pages: List of crawled pages
            
        Returns:
            Dictionary with crawling statistics
        """
        if not crawled_pages:
            return {"total_pages": 0}
        
        total_content_size = sum(len(page.html_content) for page in crawled_pages)
        avg_content_size = total_content_size / len(crawled_pages)
        
        return {
            "total_pages": len(crawled_pages),
            "base_url": self.base_url,
            "base_domain": self.base_domain,
            "total_content_size": total_content_size,
            "avg_content_size": round(avg_content_size, 2),
            "crawl_settings": {
                "max_depth": self.max_depth,
                "delay": self.delay
            },
            "pages": [
                {
                    "url": page.url,
                    "title": page.title,
                    "status_code": page.status_code,
                    "content_size": len(page.html_content),
                    "timestamp": page.crawl_timestamp
                }
                for page in crawled_pages
            ]
        }