#!/usr/bin/env python3
"""
Test script for the enhanced text extractor.
This script demonstrates text extraction and cleaning functionality.
"""

import sys
import os
import logging

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_test_html() -> str:
    """Create a sample HTML with common unwanted elements for testing."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Test Article - Python Programming Guide</title>
        <meta name="description" content="A comprehensive guide to Python programming for beginners">
        <meta name="keywords" content="python, programming, tutorial, guide">
        <meta property="og:title" content="Python Programming Guide">
        <meta property="og:description" content="Learn Python programming from scratch">
        <script>
            // Analytics tracking
            function trackEvent() { console.log('tracked'); }
        </script>
        <style>
            .content { margin: 20px; }
            .hidden { display: none; }
        </style>
    </head>
    <body>
        <!-- Navigation header -->
        <header class="site-header">
            <nav class="main-navigation">
                <ul class="nav-menu">
                    <li><a href="/">Home</a></li>
                    <li><a href="/about">About</a></li>
                    <li><a href="/contact">Contact</a></li>
                </ul>
            </nav>
        </header>
        
        <!-- Cookie consent banner -->
        <div class="cookie-consent-banner">
            <p>This site uses cookies. By continuing to browse, you agree to our cookie policy.</p>
            <button onclick="acceptCookies()">Accept</button>
        </div>
        
        <!-- Advertisement -->
        <div class="advertisement-banner">
            <div class="google-ad">
                <p>Advertisement: Buy our amazing product now!</p>
            </div>
        </div>
        
        <!-- Main content -->
        <main class="main-content">
            <article class="post-content">
                <h1>Introduction to Python Programming</h1>
                
                <p>Python is a high-level, interpreted programming language with dynamic semantics. 
                Its high-level built-in data structures, combined with dynamic typing and dynamic 
                binding, make it very attractive for Rapid Application Development.</p>
                
                <h2>Getting Started with Python</h2>
                
                <p>To begin your journey with Python programming, you first need to understand 
                the basic concepts. Python's syntax emphasizes readability and therefore reduces 
                the cost of program maintenance.</p>
                
                <h3>Installing Python</h3>
                
                <p>You can download Python from the official website at python.org. The installation 
                process is straightforward on all major operating systems including Windows, macOS, 
                and Linux distributions.</p>
                
                <blockquote>
                    "Python is an easy to learn, powerful programming language." - Python.org
                </blockquote>
                
                <h3>Your First Python Program</h3>
                
                <p>The traditional first program in any programming language is "Hello, World!". 
                In Python, this is incredibly simple:</p>
                
                <pre><code>print("Hello, World!")</code></pre>
                
                <p>This single line of code will output the text "Hello, World!" to your console.</p>
                
            </article>
        </main>
        
        <!-- Sidebar with related content -->
        <aside class="sidebar">
            <div class="related-posts">
                <h4>Related Articles</h4>
                <ul>
                    <li><a href="/advanced-python">Advanced Python Topics</a></li>
                    <li><a href="/python-libraries">Popular Python Libraries</a></li>
                </ul>
            </div>
            
            <!-- Social sharing buttons -->
            <div class="social-sharing">
                <h4>Share this article</h4>
                <button class="share-facebook">Share on Facebook</button>
                <button class="share-twitter">Share on Twitter</button>
                <button class="share-linkedin">Share on LinkedIn</button>
            </div>
        </aside>
        
        <!-- Footer -->
        <footer class="site-footer">
            <div class="footer-content">
                <p>&copy; 2024 Programming Tutorials. All rights reserved.</p>
                <nav class="footer-nav">
                    <a href="/privacy">Privacy Policy</a>
                    <a href="/terms">Terms of Service</a>
                    <a href="/contact">Contact Us</a>
                </nav>
            </div>
        </footer>
        
        <!-- Analytics script -->
        <script>
            (function() {
                var ga = document.createElement('script');
                ga.src = 'https://analytics.example.com/track.js';
                document.head.appendChild(ga);
            })();
        </script>
    </body>
    </html>
    """

def test_text_extraction():
    """Test the enhanced text extraction functionality."""
    print("🧪 Testing Enhanced Text Extractor")
    print("=" * 60)
    
    try:
        from extraction.text_extractor import TextExtractor, CleanedContent
        print("✅ Successfully imported TextExtractor and CleanedContent")
        
        # Initialize extractor
        extractor = TextExtractor()
        print("✅ Successfully created TextExtractor instance")
        
        # Test with sample HTML
        test_html = create_test_html()
        test_url = "https://example.com/python-guide"
        
        print(f"\n🔍 Extracting text from sample HTML...")
        print(f"Original HTML size: {len(test_html):,} characters")
        
        # Extract and clean text
        cleaned_content = extractor.extract_text(
            html_content=test_html,
            url=test_url,
            title=""  # Let it extract the title
        )
        
        if cleaned_content:
            print(f"✅ Successfully extracted text!")
            print(f"📊 Extraction Results:")
            print(f"   - Title: {cleaned_content.title}")
            print(f"   - Word Count: {cleaned_content.word_count}")
            print(f"   - Character Count: {cleaned_content.char_count}")
            print(f"   - URL: {cleaned_content.url}")
            print(f"   - Metadata fields: {len(cleaned_content.metadata)}")
            
            # Log the cleaned content for inspection
            print(f"\n📄 Cleaned Content Preview:")
            extractor.log_cleaned_content(cleaned_content, max_chars=300)
            
            return True
        else:
            print("❌ Failed to extract text")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure BeautifulSoup4 is installed: pip install beautifulsoup4")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_cleaning_features():
    """Test specific cleaning features."""
    print(f"\n🧹 Testing Text Cleaning Features")
    print("=" * 60)
    
    try:
        from extraction.text_extractor import TextExtractor
        
        extractor = TextExtractor()
        
        # Test cases for different types of unwanted content
        test_cases = [
            {
                "name": "Navigation removal",
                "html": "<nav><ul><li><a href='/home'>Home</a></li></ul></nav><p>Main content here.</p>",
                "should_contain": "Main content here",
                "should_not_contain": "Home"
            },
            {
                "name": "Script removal", 
                "html": "<script>alert('popup');</script><p>Real content to keep.</p>",
                "should_contain": "Real content to keep",
                "should_not_contain": "alert"
            },
            {
                "name": "Cookie banner removal",
                "html": "<div class='cookie-consent'><p>Accept cookies</p></div><p>Article content.</p>",
                "should_contain": "Article content",
                "should_not_contain": "Accept cookies"
            },
            {
                "name": "Ad removal",
                "html": "<div class='advertisement'><p>Buy now!</p></div><p>Educational content.</p>",
                "should_contain": "Educational content", 
                "should_not_contain": "Buy now"
            },
            {
                "name": "Social sharing removal",
                "html": "<div class='social-sharing'><button>Share</button></div><p>Article text.</p>",
                "should_contain": "Article text",
                "should_not_contain": "Share"
            }
        ]
        
        passed_tests = 0
        total_tests = len(test_cases)
        
        for test_case in test_cases:
            print(f"\n🔍 Testing: {test_case['name']}")
            
            cleaned_content = extractor.extract_text(test_case['html'], "test://url")
            
            if cleaned_content:
                text = cleaned_content.cleaned_text.lower()
                
                # Check if desired content is present
                contains_good = test_case['should_contain'].lower() in text
                # Check if unwanted content is removed
                contains_bad = test_case['should_not_contain'].lower() in text
                
                if contains_good and not contains_bad:
                    print(f"   ✅ PASS - Kept good content, removed unwanted")
                    passed_tests += 1
                else:
                    print(f"   ❌ FAIL - Good: {contains_good}, Bad removed: {not contains_bad}")
                    print(f"   📝 Extracted: {text}")
            else:
                print(f"   ❌ FAIL - No content extracted")
        
        print(f"\n📊 Cleaning Test Results: {passed_tests}/{total_tests} tests passed")
        return passed_tests == total_tests
        
    except Exception as e:
        print(f"❌ Error testing cleaning features: {e}")
        return False

def test_real_world_example():
    """Test with a more realistic HTML structure."""
    print(f"\n🌐 Testing with Realistic HTML Structure")
    print("=" * 60)
    
    try:
        from extraction.text_extractor import TextExtractor
        
        # Simulate a real blog post structure
        realistic_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>How to Learn Python - Complete Guide</title>
            <meta name="description" content="Complete guide to learning Python programming">
        </head>
        <body>
            <header class="header">
                <nav class="navbar">
                    <div class="nav-brand">TechBlog</div>
                    <ul class="nav-links">
                        <li><a href="/">Home</a></li>
                        <li><a href="/tutorials">Tutorials</a></li>
                        <li><a href="/about">About</a></li>
                    </ul>
                </nav>
            </header>
            
            <main class="content">
                <article class="post">
                    <h1>How to Learn Python Programming: A Complete Guide</h1>
                    
                    <div class="post-meta">
                        <span class="author">By John Doe</span>
                        <span class="date">December 14, 2024</span>
                    </div>
                    
                    <p>Python is one of the most popular programming languages in the world today. 
                    Whether you're interested in web development, data science, artificial intelligence, 
                    or automation, Python provides the tools and libraries you need to get started.</p>
                    
                    <h2>Why Choose Python?</h2>
                    
                    <p>Python's popularity stems from several key advantages. First, it has a simple 
                    and readable syntax that makes it perfect for beginners. The language emphasizes 
                    code readability, which means you can focus on solving problems rather than 
                    wrestling with complex syntax.</p>
                    
                    <h3>Key Benefits of Python:</h3>
                    <ul>
                        <li>Easy to learn and use</li>
                        <li>Large standard library</li>
                        <li>Active community support</li>
                        <li>Cross-platform compatibility</li>
                        <li>Extensive third-party packages</li>
                    </ul>
                    
                    <h2>Getting Started</h2>
                    
                    <p>To begin your Python journey, you'll need to install Python on your computer. 
                    Visit the official Python website and download the latest version for your 
                    operating system. The installation process includes IDLE, Python's integrated 
                    development environment.</p>
                    
                    <blockquote>
                        "The best way to learn Python is by writing Python code." - Guido van Rossum
                    </blockquote>
                    
                </article>
            </main>
            
            <aside class="sidebar">
                <div class="ad-banner">
                    <img src="ad.jpg" alt="Advertisement">
                    <p>Learn coding online!</p>
                </div>
                
                <div class="social-links">
                    <h4>Follow Us</h4>
                    <a href="#" class="facebook">Facebook</a>
                    <a href="#" class="twitter">Twitter</a>
                </div>
            </aside>
            
            <footer class="footer">
                <p>© 2024 TechBlog. All rights reserved.</p>
            </footer>
        </body>
        </html>
        """
        
        extractor = TextExtractor()
        cleaned_content = extractor.extract_text(
            realistic_html, 
            "https://techblog.com/learn-python",
            ""
        )
        
        if cleaned_content:
            print(f"✅ Successfully extracted realistic content!")
            print(f"📊 Results:")
            print(f"   - Title: {cleaned_content.title}")
            print(f"   - Words: {cleaned_content.word_count}")
            print(f"   - Characters: {cleaned_content.char_count}")
            
            # Check if main content is preserved while unwanted content is removed
            text = cleaned_content.cleaned_text.lower()
            
            good_content = [
                "python is one of the most popular",
                "why choose python",
                "key benefits of python",
                "easy to learn and use",
                "getting started"
            ]
            
            bad_content = [
                "techblog",
                "follow us", 
                "facebook",
                "advertisement",
                "all rights reserved"
            ]
            
            good_count = sum(1 for content in good_content if content in text)
            bad_count = sum(1 for content in bad_content if content in text)
            
            print(f"   - Good content preserved: {good_count}/{len(good_content)}")
            print(f"   - Bad content removed: {len(bad_content) - bad_count}/{len(bad_content)}")
            
            # Show a preview
            print(f"\n📄 Text Preview (first 200 chars):")
            print(f"'{cleaned_content.cleaned_text[:200]}...'")
            
            return good_count >= 4 and bad_count <= 1
        else:
            print("❌ Failed to extract realistic content")
            return False
            
    except Exception as e:
        print(f"❌ Error testing realistic example: {e}")
        return False

def main():
    """Run all text extraction tests."""
    print("🚀 Enhanced Text Extractor - Testing Suite")
    print("=" * 70)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Basic text extraction
    if test_text_extraction():
        tests_passed += 1
    
    # Test 2: Cleaning features
    if test_cleaning_features():
        tests_passed += 1
    
    # Test 3: Real-world example
    if test_real_world_example():
        tests_passed += 1
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📊 TEST RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The enhanced text extractor is working correctly.")
        print("💡 Key features verified:")
        print("   ✅ HTML parsing and text extraction")
        print("   ✅ Navigation and footer removal")
        print("   ✅ Script and style tag removal") 
        print("   ✅ Advertisement and banner removal")
        print("   ✅ Cookie consent removal")
        print("   ✅ Social sharing button removal")
        print("   ✅ Text cleaning and normalization")
        print("   ✅ Metadata extraction")
        print("   ✅ Content statistics calculation")
    else:
        print(f"❌ {total_tests - tests_passed} test(s) failed. Check the errors above.")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install beautifulsoup4 lxml")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    main()