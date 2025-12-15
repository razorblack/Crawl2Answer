"""
Text extraction module for cleaning HTML and extracting readable content.
"""

from bs4 import BeautifulSoup, Comment
import re
import logging
from typing import Optional, Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CleanedContent:
    """Data structure for storing cleaned content."""
    url: str
    title: str
    cleaned_text: str
    word_count: int
    char_count: int
    extraction_timestamp: str
    metadata: Dict


class TextExtractor:
    """Enhanced text extractor for cleaning HTML and extracting readable content."""
    
    def __init__(self):
        """Initialize the text extractor with comprehensive cleaning rules."""
        
        # Tags to completely remove (including their content)
        self.remove_tags = [
            'script', 'style', 'noscript', 'iframe', 'embed', 'object',
            'applet', 'canvas', 'svg', 'math', 'select', 'option',
            'textarea', 'input', 'button', 'form'
        ]
        
        # Navigation and structural tags to remove
        self.nav_tags = [
            'nav', 'header', 'footer', 'aside', 'menu', 'menuitem',
            'toolbar', 'breadcrumb'
        ]
        
        # Advertisement and tracking tags
        self.ad_tags = [
            'advertisement', 'ad', 'ads', 'google-ad', 'doubleclick',
            'adsense', 'adsbygoogle', 'sponsored'
        ]
        
        # Common class/id patterns for unwanted content
        self.unwanted_patterns = [
            # Navigation
            r'nav(?:igation)?', r'menu', r'breadcrumb', r'sidebar',
            r'header', r'footer', r'top-?bar', r'bottom-?bar',
            
            # Advertisements
            r'ad(?:vertisement)?s?', r'banner', r'promo', r'sponsor',
            r'google-?ad', r'adsense', r'doubleclick',
            
            # Social and sharing
            r'social', r'share', r'sharing', r'follow', r'subscribe',
            r'facebook', r'twitter', r'linkedin', r'pinterest',
            
            # Comments and user content
            r'comment', r'review', r'rating', r'feedback',
            
            # Cookie and privacy
            r'cookie', r'privacy', r'consent', r'gdpr', r'ccpa',
            
            # Pagination and controls
            r'pag(?:ination|ing)', r'page-?nav', r'next', r'prev',
            r'controls?', r'buttons?',
            
            # Metadata and tags
            r'tags?', r'categories', r'meta(?:data)?', r'related',
            
            # Utility
            r'print', r'email', r'search', r'filter', r'sort'
        ]
        
        # Compile regex patterns for better performance
        self.unwanted_regex = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.unwanted_patterns
        ]
    
    def extract_text(self, html_content: str, url: str = "", title: str = "") -> Optional[CleanedContent]:
        """
        Extract and clean text from HTML content.
        
        Args:
            html_content: Raw HTML content
            url: Source URL (optional)
            title: Page title (optional)
            
        Returns:
            CleanedContent object with cleaned text and metadata, or None if extraction fails
        """
        try:
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title if not provided
            if not title:
                title = self._extract_title(soup)
            
            # Step 1: Remove unwanted tags completely
            self._remove_unwanted_tags(soup)
            
            # Step 2: Remove elements by class/id patterns
            self._remove_by_patterns(soup)
            
            # Step 3: Remove comments and processing instructions
            self._remove_comments(soup)
            
            # Step 4: Extract main content area if possible
            main_content = self._extract_main_content(soup)
            
            # Step 5: Extract and clean text
            raw_text = main_content.get_text(separator=' ', strip=True) if main_content else ""
            
            # Step 6: Clean and normalize text
            cleaned_text = self._clean_text(raw_text)
            
            if not cleaned_text or len(cleaned_text.strip()) < 50:
                logger.warning(f"Extracted text too short or empty for {url}")
                return None
            
            # Step 7: Calculate statistics
            word_count = len(cleaned_text.split())
            char_count = len(cleaned_text)
            
            # Step 8: Extract additional metadata
            metadata = self._extract_enhanced_metadata(soup, html_content)
            
            # Step 9: Create timestamp
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            
            # Create CleanedContent object
            cleaned_content = CleanedContent(
                url=url,
                title=title,
                cleaned_text=cleaned_text,
                word_count=word_count,
                char_count=char_count,
                extraction_timestamp=timestamp,
                metadata=metadata
            )
            
            logger.info(f"Successfully extracted {word_count} words from {url}")
            return cleaned_content
            
        except Exception as e:
            logger.error(f"Failed to extract text from {url}: {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract title from HTML."""
        # Try title tag first
        title_tag = soup.find('title')
        if title_tag and title_tag.get_text().strip():
            return title_tag.get_text().strip()
        
        # Try h1 tag
        h1_tag = soup.find('h1')
        if h1_tag and h1_tag.get_text().strip():
            return h1_tag.get_text().strip()
        
        # Try og:title meta tag
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            return og_title.get('content').strip()
        
        return "Untitled Page"
    
    def _remove_unwanted_tags(self, soup: BeautifulSoup):
        """Remove unwanted tags and their content."""
        all_unwanted = self.remove_tags + self.nav_tags + self.ad_tags
        
        for tag_name in all_unwanted:
            for tag in soup.find_all(tag_name):
                tag.decompose()
    
    def _remove_by_patterns(self, soup: BeautifulSoup):
        """Remove elements based on class/id patterns."""
        for element in soup.find_all(True):  # Find all tags
            # Check class attribute
            class_attr = element.get('class', [])
            if isinstance(class_attr, list):
                class_str = ' '.join(class_attr).lower()
            else:
                class_str = str(class_attr).lower()
            
            # Check id attribute
            id_attr = element.get('id', '').lower()
            
            # Check if any pattern matches
            for regex in self.unwanted_regex:
                if regex.search(class_str) or regex.search(id_attr):
                    element.decompose()
                    break
    
    def _remove_comments(self, soup: BeautifulSoup):
        """Remove HTML comments and processing instructions."""
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
    
    def _extract_main_content(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Try to identify and extract the main content area."""
        # Common patterns for main content
        main_selectors = [
            'main', 'article', '[role="main"]',
            '.main-content', '.content', '.post-content',
            '.article-content', '.entry-content', '.page-content',
            '#main-content', '#content', '#post-content',
            '#article-content', '#entry-content'
        ]
        
        for selector in main_selectors:
            try:
                main_element = soup.select_one(selector)
                if main_element and len(main_element.get_text(strip=True)) > 200:
                    logger.debug(f"Found main content using selector: {selector}")
                    return main_element
            except:
                continue
        
        # Fallback: try to find the largest text container
        text_containers = soup.find_all(['div', 'section', 'article'])
        if text_containers:
            largest = max(text_containers, key=lambda x: len(x.get_text(strip=True)))
            if len(largest.get_text(strip=True)) > 200:
                return largest
        
        # Last resort: return the body or the whole soup
        body = soup.find('body')
        return body if body else soup
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Step 1: Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Step 2: Remove excessive punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\'\"]+', ' ', text)
        
        # Step 3: Fix common issues
        text = re.sub(r'\s+([\.,:;!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([\.!?])\s*([A-Z])', r'\1 \2', text)  # Ensure space after sentence endings
        
        # Step 4: Remove URLs and email addresses
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        
        # Step 5: Remove excessive repetition
        text = re.sub(r'(.)\1{3,}', r'\1\1', text)  # Reduce repeated characters
        
        # Step 6: Clean up lines
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip very short lines or lines with mostly non-alphanumeric characters
            if len(line) > 10 and sum(c.isalnum() for c in line) / len(line) > 0.5:
                cleaned_lines.append(line)
        
        # Step 7: Join lines and final cleanup
        cleaned_text = ' '.join(cleaned_lines)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Final whitespace cleanup
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    def _extract_enhanced_metadata(self, soup: BeautifulSoup, html_content: str) -> Dict:
        """Extract enhanced metadata from HTML."""
        metadata = {}
        
        try:
            # Basic meta tags
            description = soup.find('meta', attrs={'name': 'description'})
            if description:
                metadata['description'] = description.get('content', '').strip()
            
            keywords = soup.find('meta', attrs={'name': 'keywords'})
            if keywords:
                metadata['keywords'] = keywords.get('content', '').strip()
            
            # Open Graph tags
            og_tags = ['og:title', 'og:description', 'og:type', 'og:url', 'og:image']
            for og_tag in og_tags:
                tag = soup.find('meta', property=og_tag)
                if tag:
                    metadata[og_tag.replace(':', '_')] = tag.get('content', '').strip()
            
            # Language
            html_tag = soup.find('html')
            if html_tag:
                lang = html_tag.get('lang')
                if lang:
                    metadata['language'] = lang.strip()
            
            # Headings structure
            headings = []
            for level in range(1, 7):
                h_tags = soup.find_all(f'h{level}')
                for h_tag in h_tags:
                    heading_text = h_tag.get_text().strip()
                    if heading_text:
                        headings.append({
                            'level': level,
                            'text': heading_text
                        })
            metadata['headings'] = headings
            
            # Links count
            links = soup.find_all('a', href=True)
            metadata['links_count'] = len(links)
            
            # Images count
            images = soup.find_all('img', src=True)
            metadata['images_count'] = len(images)
            
            # Content statistics
            metadata['html_size'] = len(html_content)
            
        except Exception as e:
            logger.warning(f"Failed to extract some metadata: {e}")
        
        return metadata
    
    def extract_metadata(self, html_content: str) -> Dict:
        """
        Extract metadata from HTML content (backwards compatibility).
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            Dictionary containing metadata
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            return self._extract_enhanced_metadata(soup, html_content)
        except Exception as e:
            logger.error(f"Failed to extract metadata: {e}")
            return {}
    
    def log_cleaned_content(self, cleaned_content: CleanedContent, max_chars: int = 500):
        """
        Log cleaned content for testing purposes.
        
        Args:
            cleaned_content: CleanedContent object
            max_chars: Maximum characters to display in log
        """
        logger.info("=" * 80)
        logger.info(f"CLEANED CONTENT PREVIEW")
        logger.info("=" * 80)
        logger.info(f"URL: {cleaned_content.url}")
        logger.info(f"Title: {cleaned_content.title}")
        logger.info(f"Word Count: {cleaned_content.word_count}")
        logger.info(f"Character Count: {cleaned_content.char_count}")
        logger.info(f"Extracted: {cleaned_content.extraction_timestamp}")
        logger.info("-" * 80)
        
        # Show preview of cleaned text
        preview_text = cleaned_content.cleaned_text
        if len(preview_text) > max_chars:
            preview_text = preview_text[:max_chars] + "..."
        
        logger.info(f"CLEANED TEXT PREVIEW:")
        logger.info(preview_text)
        
        # Show metadata
        if cleaned_content.metadata:
            logger.info("-" * 80)
            logger.info("METADATA:")
            for key, value in cleaned_content.metadata.items():
                if isinstance(value, list) and len(value) > 3:
                    logger.info(f"  {key}: {value[:3]}... ({len(value)} total)")
                else:
                    value_str = str(value)[:100]
                    logger.info(f"  {key}: {value_str}")
        
        logger.info("=" * 80)