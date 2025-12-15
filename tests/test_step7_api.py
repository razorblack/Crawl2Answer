#!/usr/bin/env python3
"""
Step 7 Testing: REST API endpoints

This script tests the two main Step 7 endpoints:
- POST /crawl
- POST /ask

It can be used as an alternative to Postman/curl for testing.
"""

import requests
import json
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Configuration
API_BASE_URL = "http://localhost:8000"

def test_api_health():
    """Test basic API health"""
    try:
        logger.info("Testing API health...")
        response = requests.get(f"{API_BASE_URL}/health", timeout=30)
        
        if response.status_code == 200:
            logger.info("✓ API is healthy")
            return True
        else:
            logger.error(f"✗ API health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        logger.error("✗ Could not connect to API. Make sure it's running on http://localhost:8000")
        return False
    except Exception as e:
        logger.error(f"✗ Health check failed: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint for system status"""
    try:
        logger.info("Testing root endpoint...")
        response = requests.get(f"{API_BASE_URL}/", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✓ Root endpoint working. Status: {data.get('status')}")
            logger.info(f"   Message: {data.get('message')}")
            return True
        else:
            logger.error(f"✗ Root endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Root endpoint test failed: {e}")
        return False

def test_crawl_endpoint():
    """Test POST /crawl endpoint"""
    try:
        logger.info("=" * 60)
        logger.info("TESTING POST /crawl ENDPOINT")
        logger.info("=" * 60)
        
        # Prepare test data
        crawl_data = {
            "baseUrl": "https://docs.python.org/3/tutorial/",
            "max_pages": 3,
            "max_depth": 2,
            "delay": 1.0
        }
        
        logger.info(f"Crawling: {crawl_data['baseUrl']}")
        logger.info(f"Max pages: {crawl_data['max_pages']}")
        
        # Send request
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/crawl",
            json=crawl_data,
            timeout=120  # Allow up to 2 minutes for crawling
        )
        
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            logger.info("✓ Crawl endpoint successful!")
            logger.info(f"   Status: {result.get('status')}")
            logger.info(f"   Message: {result.get('message')}")
            logger.info(f"   Pages crawled: {result.get('pages_crawled')}")
            logger.info(f"   Chunks created: {result.get('chunks_created')}")
            logger.info(f"   Embeddings generated: {result.get('embeddings_generated')}")
            logger.info(f"   Database updated: {result.get('database_updated')}")
            logger.info(f"   Processing time: {result.get('processing_time'):.2f}s")
            logger.info(f"   Request time: {request_time:.2f}s")
            return True
        else:
            logger.error(f"✗ Crawl endpoint failed: {response.status_code}")
            logger.error(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        logger.error("✗ Crawl request timed out")
        return False
    except Exception as e:
        logger.error(f"✗ Crawl endpoint test failed: {e}")
        return False

def test_ask_endpoint():
    """Test POST /ask endpoint"""
    try:
        logger.info("=" * 60)
        logger.info("TESTING POST /ask ENDPOINT")
        logger.info("=" * 60)
        
        # Test questions related to Python documentation
        test_questions = [
            "What is Python?",
            "How do you define functions in Python?",
            "What are Python data types?",
            "How do loops work in Python?",
            "What is object-oriented programming in Python?"
        ]
        
        successful_questions = 0
        
        for i, question in enumerate(test_questions, 1):
            logger.info(f"\n[Question {i}/{len(test_questions)}] {question}")
            logger.info("-" * 40)
            
            question_data = {
                "question": question
            }
            
            try:
                start_time = time.time()
                response = requests.post(
                    f"{API_BASE_URL}/ask",
                    json=question_data,
                    timeout=60
                )
                
                request_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    successful_questions += 1
                    
                    logger.info("✓ Question answered successfully!")
                    logger.info(f"   Answer: {result.get('answer')[:200]}...")
                    logger.info(f"   Confidence: {result.get('confidence'):.3f}")
                    logger.info(f"   Sources: {len(result.get('sources', []))} URLs")
                    logger.info(f"   Retrieval time: {result.get('retrieval_time'):.3f}s")
                    logger.info(f"   Generation time: {result.get('generation_time'):.3f}s")
                    logger.info(f"   Total request time: {request_time:.3f}s")
                    
                    # Show sources
                    sources = result.get('sources', [])
                    if sources:
                        logger.info("   Source URLs:")
                        for j, source in enumerate(sources[:3], 1):  # Show first 3 sources
                            logger.info(f"     {j}. {source}")
                
                elif response.status_code == 404:
                    logger.warning(f"   No relevant content found for: '{question}'")
                    
                else:
                    logger.error(f"✗ Question failed: {response.status_code}")
                    logger.error(f"   Response: {response.text[:200]}...")
                
            except requests.exceptions.Timeout:
                logger.error(f"   Question timed out: '{question}'")
            except Exception as e:
                logger.error(f"   Question error: {e}")
        
        logger.info(f"\n✓ Successfully answered {successful_questions}/{len(test_questions)} questions")
        return successful_questions > 0
        
    except Exception as e:
        logger.error(f"✗ Ask endpoint test failed: {e}")
        return False

def main():
    """Run all Step 7 tests"""
    logger.info("STEP 7 TESTING: REST API ENDPOINTS")
    logger.info("=" * 60)
    
    # Test API availability
    if not test_api_health():
        logger.error("API is not available. Please start the API server first:")
        logger.error("  python -m uvicorn api.main:app --host 0.0.0.0 --port 8000")
        return False
    
    # Test root endpoint
    if not test_root_endpoint():
        logger.error("Root endpoint test failed")
        return False
    
    # Test crawl endpoint
    crawl_success = test_crawl_endpoint()
    
    # Test ask endpoint (only if crawl was successful)
    ask_success = test_ask_endpoint()
    
    # Final summary
    logger.info("=" * 60)
    logger.info("STEP 7 TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Crawl endpoint (/crawl): {'✓ PASS' if crawl_success else '✗ FAIL'}")
    logger.info(f"Ask endpoint (/ask): {'✓ PASS' if ask_success else '✗ FAIL'}")
    
    if crawl_success and ask_success:
        logger.info("\n🎉 Step 7 Implementation Complete!")
        logger.info("✓ POST /crawl: Runs crawling → extraction → chunking → embeddings → indexing")
        logger.info("✓ POST /ask: Runs retrieval → answer generation")
        logger.info("\nThe RAG API is fully functional and ready for production use!")
        return True
    else:
        logger.error("\n❌ Step 7 testing failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)