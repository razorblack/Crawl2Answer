"""
Configuration settings for the Crawl2Answer application.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""
    
    def __init__(self):
        """Initialize settings from environment variables."""
        
        # Basic configuration
        self.PROJECT_NAME = "Crawl2Answer"
        self.VERSION = "1.0.0"
        self.DEBUG = self._get_bool_env("DEBUG", False)
        
        # API configuration
        self.API_HOST = os.getenv("API_HOST", "0.0.0.0")
        self.API_PORT = int(os.getenv("API_PORT", "8000"))
        
        # Website crawling configuration
        self.BASE_URL = os.getenv("BASE_URL", "")
        self.CRAWL_DELAY = float(os.getenv("CRAWL_DELAY", "1.0"))
        self.MAX_PAGES = int(os.getenv("MAX_PAGES", "10"))
        self.MAX_DEPTH = int(os.getenv("MAX_DEPTH", "3"))
        self.REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
        
        # Text processing configuration
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
        
        # Embedding configuration
        self.EMBEDDING_MODEL_TYPE = os.getenv("EMBEDDING_MODEL_TYPE", "sentence_transformers")
        self.EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        
        # Vector database configuration
        self.VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "data/embeddings")
        
        # Retrieval configuration
        self.RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))
        self.SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.1"))
        
        # Logging configuration
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FILE = os.getenv("LOG_FILE", "")
        
        # Data paths
        self.DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
        self.RAW_DATA_DIR = self.DATA_DIR / "raw"
        self.PROCESSED_DATA_DIR = self.DATA_DIR / "processed"
        self.EMBEDDINGS_DIR = self.DATA_DIR / "embeddings"
        
        # Create directories if they don't exist
        self._create_directories()
        
        # Validate configuration
        self._validate_settings()
    
    def _get_bool_env(self, key: str, default: bool = False) -> bool:
        """Get boolean value from environment variable."""
        value = os.getenv(key, "").lower()
        return value in ("true", "1", "yes", "on") if value else default
    
    def _create_directories(self):
        """Create necessary directories."""
        for directory in [self.DATA_DIR, self.RAW_DATA_DIR, 
                         self.PROCESSED_DATA_DIR, self.EMBEDDINGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _validate_settings(self):
        """Validate configuration settings."""
        if self.EMBEDDING_MODEL_TYPE == "openai" and not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when using OpenAI embeddings")
        
        if self.CHUNK_SIZE <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        
        if self.RETRIEVAL_K <= 0:
            raise ValueError("RETRIEVAL_K must be positive")
    
    def get_database_url(self) -> str:
        """Get database URL for vector storage."""
        return str(self.EMBEDDINGS_DIR)
    
    def to_dict(self) -> dict:
        """Convert settings to dictionary."""
        return {
            key: getattr(self, key)
            for key in dir(self)
            if not key.startswith('_') and not callable(getattr(self, key))
        }