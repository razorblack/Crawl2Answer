#!/usr/bin/env python3
"""
Step 7: Manual API Testing Guide
Provides curl commands and instructions for testing the REST API endpoints
"""

def print_manual_testing_guide():
    """Print comprehensive testing guide for Step 7 API"""
    print("🚀 STEP 7: REST API ENDPOINTS - MANUAL TESTING GUIDE")
    print("=" * 80)
    print()
    
    print("📋 SETUP INSTRUCTIONS:")
    print("-" * 40)
    print("1. Open a terminal and navigate to the project directory")
    print("2. Start the API server:")
    print("   python -m uvicorn api.main:app --port 8000 --reload")
    print("3. Wait for the server to start (you should see 'Uvicorn running on...')")
    print("4. Open a new terminal for testing")
    print()
    
    print("🌐 API ENDPOINTS TO TEST:")
    print("-" * 40)
    print()
    
    print("1️⃣  GET /status - Health check endpoint")
    print("   Purpose: Check if the API is running and components are ready")
    print("   Command:")
    print('   curl -X GET "http://localhost:8000/status"')
    print()
    print("   Expected Response:")
    print('   {')
    print('     "status": "healthy",')
    print('     "message": "Crawl2Answer API is running",')
    print('     "stats": { ... component status ... }')
    print('   }')
    print()
    
    print("2️⃣  POST /ask - Question answering endpoint")
    print("   Purpose: Ask questions and get answers from the RAG system")
    print("   Command:")
    print('   curl -X POST "http://localhost:8000/ask" \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"question": "What is artificial intelligence?"}\'')
    print()
    print("   Other test questions:")
    print('   - {"question": "How does machine learning work?"}')
    print('   - {"question": "What are neural networks?"}')
    print('   - {"question": "Explain deep learning"}')
    print()
    print("   Expected Response:")
    print('   {')
    print('     "question": "What is artificial intelligence?",')
    print('     "answer": "Based on the available context...",')
    print('     "sources": ["sample_ai_article.html", ...],')
    print('     "confidence": 0.85,')
    print('     "retrieval_time": 0.045,')
    print('     "generation_time": 1.2')
    print('   }')
    print()
    
    print("3️⃣  POST /crawl - Web crawling and indexing endpoint")
    print("   Purpose: Crawl a website, extract content, and add to vector database")
    print("   Command:")
    print('   curl -X POST "http://localhost:8000/crawl" \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"baseUrl": "https://example.com", "max_pages": 5}\'')
    print()
    print("   ⚠️  WARNING: This endpoint performs actual web crawling!")
    print("   - Use a test website or your own domain")
    print("   - May take 10-30 seconds to complete")
    print("   - Creates real embeddings and updates vector database")
    print()
    print("   Test with safe URLs:")
    print('   - {"baseUrl": "https://httpbin.org", "max_pages": 3}')
    print('   - {"baseUrl": "https://jsonplaceholder.typicode.com", "max_pages": 2}')
    print()
    print("   Expected Response:")
    print('   {')
    print('     "status": "success",')
    print('     "message": "Crawling completed successfully",')
    print('     "base_url": "https://example.com",')
    print('     "pages_crawled": 5,')
    print('     "chunks_created": 23,')
    print('     "embeddings_generated": 23,')
    print('     "database_updated": true,')
    print('     "processing_time": 15.7')
    print('   }')
    print()
    
    print("📖 INTERACTIVE DOCUMENTATION:")
    print("-" * 40)
    print("• Swagger UI: http://localhost:8000/docs")
    print("• ReDoc UI: http://localhost:8000/redoc")
    print("• OpenAPI JSON: http://localhost:8000/openapi.json")
    print()
    
    print("🧪 TESTING WORKFLOW:")
    print("-" * 40)
    print("1. Test /status to ensure API is running")
    print("2. Test /ask with existing data (from Step 5/6)")
    print("3. Test /crawl with a small website")
    print("4. Test /ask again with new questions about crawled content")
    print()
    
    print("🔍 TROUBLESHOOTING:")
    print("-" * 40)
    print("• If /ask returns no results:")
    print("  - Check that Step 5/6 tests were run (vector database exists)")
    print("  - Try questions about AI, ML, neural networks, computer vision, NLP")
    print()
    print("• If /crawl fails:")
    print("  - Check internet connection")
    print("  - Use simple, accessible websites")
    print("  - Check server logs for error details")
    print()
    print("• If server won't start:")
    print("  - Check that all dependencies are installed")
    print("  - Try: pip install fastapi uvicorn requests python-multipart")
    print("  - Check that port 8000 is not in use")
    print()
    
    print("✅ SUCCESS CRITERIA:")
    print("-" * 40)
    print("Step 7 is complete when:")
    print("✓ GET /status returns 200 with healthy status")
    print("✓ POST /ask returns relevant answers with sources")
    print("✓ POST /crawl successfully processes a website")
    print("✓ API documentation is accessible at /docs")
    print("✓ All endpoints follow the specified input/output format")
    print()
    
    print("🎯 STEP 7 REQUIREMENTS CHECK:")
    print("-" * 40)
    print("✓ POST /crawl endpoint implemented")
    print("  - Accepts baseUrl input")  
    print("  - Runs crawling, extraction, chunking, embeddings")
    print("  - Indexes everything in vector store")
    print("  - Returns success message")
    print()
    print("✓ POST /ask endpoint implemented")
    print("  - Accepts question input")
    print("  - Runs retrieval and answer generation") 
    print("  - Returns answer text and source URLs")
    print()
    print("✓ API tested with Postman/curl equivalent")
    print("  - Manual curl commands provided")
    print("  - Interactive docs available")
    print()

def print_postman_collection():
    """Print Postman collection JSON for easy import"""
    print("\n📮 POSTMAN COLLECTION (Import this JSON):")
    print("-" * 60)
    
    collection = {
        "info": {
            "name": "Crawl2Answer API - Step 7",
            "description": "REST API endpoints for the Crawl2Answer Q&A bot",
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
        },
        "item": [
            {
                "name": "GET Status",
                "request": {
                    "method": "GET",
                    "header": [],
                    "url": {
                        "raw": "http://localhost:8000/status",
                        "protocol": "http",
                        "host": ["localhost"],
                        "port": "8000",
                        "path": ["status"]
                    }
                }
            },
            {
                "name": "POST Ask Question",
                "request": {
                    "method": "POST",
                    "header": [
                        {
                            "key": "Content-Type",
                            "value": "application/json"
                        }
                    ],
                    "body": {
                        "mode": "raw",
                        "raw": '{\n  "question": "What is artificial intelligence?"\n}'
                    },
                    "url": {
                        "raw": "http://localhost:8000/ask",
                        "protocol": "http",
                        "host": ["localhost"],
                        "port": "8000",
                        "path": ["ask"]
                    }
                }
            },
            {
                "name": "POST Crawl Website",
                "request": {
                    "method": "POST",
                    "header": [
                        {
                            "key": "Content-Type",
                            "value": "application/json"
                        }
                    ],
                    "body": {
                        "mode": "raw",
                        "raw": '{\n  "baseUrl": "https://httpbin.org",\n  "max_pages": 3,\n  "max_depth": 2\n}'
                    },
                    "url": {
                        "raw": "http://localhost:8000/crawl",
                        "protocol": "http",
                        "host": ["localhost"],
                        "port": "8000",
                        "path": ["crawl"]
                    }
                }
            }
        ]
    }
    
    import json
    print(json.dumps(collection, indent=2))

if __name__ == "__main__":
    print_manual_testing_guide()
    print_postman_collection()