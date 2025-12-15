#!/usr/bin/env python3
"""
Step 7 Full API Testing with actual HTTP requests
Tests the complete API including the actual FastAPI endpoints
"""

import requests
import json
import time
import subprocess
import threading
import signal
import os
import sys
from typing import Optional

def start_server_background() -> Optional[subprocess.Popen]:
    """Start the API server in background"""
    try:
        print("🚀 Starting API server in background...")
        
        # Start the server process
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api.main:app", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd()
        )
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ Server started successfully")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Server failed to start:")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return None

def test_server_health(base_url: str = "http://localhost:8000") -> bool:
    """Test if the server is responding"""
    try:
        response = requests.get(f"{base_url}/status", timeout=5)
        return response.status_code == 200
    except:
        return False

def test_status_endpoint(base_url: str = "http://localhost:8000") -> bool:
    """Test the GET /status endpoint"""
    print("\n" + "="*60)
    print("TESTING GET /status ENDPOINT")
    print("="*60)
    
    try:
        response = requests.get(f"{base_url}/status")
        
        print(f"📥 Request: GET {base_url}/status")
        print(f"📤 Response Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"📦 Response Data:")
            print(f"  Status: {data.get('status')}")
            print(f"  Message: {data.get('message')}")
            print(f"  Stats: {data.get('stats')}")
            print("✅ Status endpoint test PASSED")
            return True
        else:
            print(f"❌ Status endpoint test FAILED: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Status endpoint test FAILED: {e}")
        return False

def test_ask_endpoint(base_url: str = "http://localhost:8000") -> bool:
    """Test the POST /ask endpoint"""
    print("\n" + "="*60)
    print("TESTING POST /ask ENDPOINT")
    print("="*60)
    
    try:
        # Test data
        test_question = "What is artificial intelligence?"
        payload = {"question": test_question}
        
        print(f"📥 Request: POST {base_url}/ask")
        print(f"📦 Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(
            f"{base_url}/ask",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"📤 Response Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"📦 Response Data:")
            print(f"  Question: {data.get('question')}")
            print(f"  Answer: {data.get('answer', 'No answer')[:200]}...")
            print(f"  Sources: {data.get('sources', [])}")
            print(f"  Confidence: {data.get('confidence')}")
            print(f"  Retrieval Time: {data.get('retrieval_time')}s")
            print(f"  Generation Time: {data.get('generation_time')}s")
            print("✅ Ask endpoint test PASSED")
            return True
        else:
            print(f"❌ Ask endpoint test FAILED: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Ask endpoint test FAILED: {e}")
        return False

def create_curl_examples():
    """Create curl examples for manual testing"""
    print("\n" + "="*80)
    print("CURL EXAMPLES FOR MANUAL TESTING")
    print("="*80)
    
    print("\n1. Test Status Endpoint:")
    print('curl -X GET "http://localhost:8000/status"')
    
    print("\n2. Test Ask Endpoint:")
    print('curl -X POST "http://localhost:8000/ask" \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"question": "What is machine learning?"}\'')
    
    print("\n3. Test Crawl Endpoint:")
    print('curl -X POST "http://localhost:8000/crawl" \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"baseUrl": "https://example.com", "max_pages": 5}\'')
    
    print("\n4. API Documentation:")
    print("Visit: http://localhost:8000/docs")

def main():
    """Main testing function"""
    print("🚀 STEP 7 - REST API ENDPOINT TESTING")
    print("=" * 80)
    
    base_url = "http://localhost:8000"
    
    # Check if server is already running
    if test_server_health(base_url):
        print("✅ Server is already running")
        server_process = None
    else:
        print("🔄 Server not detected, attempting to start...")
        server_process = start_server_background()
        
        if not server_process:
            print("❌ Could not start server automatically")
            print("\nPlease start the server manually:")
            print("python -m uvicorn api.main:app --port 8000")
            print("\nThen run this test again or use the curl examples below:")
            create_curl_examples()
            return False
        
        # Wait for server to be ready
        print("⏳ Waiting for server to be ready...")
        for i in range(10):
            if test_server_health(base_url):
                print("✅ Server is ready")
                break
            time.sleep(1)
        else:
            print("❌ Server failed to become ready")
            return False
    
    try:
        # Run all endpoint tests
        results = []
        results.append(test_status_endpoint(base_url))
        results.append(test_ask_endpoint(base_url))  
        
        # Summary
        print("\n" + "="*80)
        print("API TESTING SUMMARY")
        print("="*80)
        
        passed = sum(results)
        total = len(results)
        
        print(f"Tests Passed: {passed}/{total}")
        
        endpoints = ["Status", "Ask"]
        for endpoint, result in zip(endpoints, results):
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {endpoint} Endpoint: {status}")
        
        if passed == total:
            print("\n🎉 All API tests passed!")
            print(f"✨ API is running at: {base_url}")
            print(f"📖 Documentation: {base_url}/docs")
            print(f"🔧 Alternative docs: {base_url}/redoc")
        else:
            print(f"\n❌ {total-passed} test(s) failed")
        
        # Show curl examples
        create_curl_examples()
        
        return passed == total
        
    finally:
        # Cleanup
        if server_process:
            print(f"\n🛑 Stopping test server...")
            server_process.terminate()
            server_process.wait()

if __name__ == "__main__":
    main()