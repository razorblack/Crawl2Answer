#!/usr/bin/env python3
"""
Step 7 API Server Startup Script
"""

import uvicorn
from api.main import app

if __name__ == "__main__":
    print("🚀 Starting Crawl2Answer API Server...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔧 Interactive Docs: http://localhost:8000/redoc")
    print("⚡ Health Check: http://localhost:8000/status")
    print("-" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )