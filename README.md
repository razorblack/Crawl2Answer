# Crawl2Answer

**Crawl. Retrieve. Answer.**

A Q&A support bot using Retrieval Augmented Generation (RAG) that crawls websites, extracts clean textual content, and provides accurate answers based on retrieved information.

## 🎯 Overview

Crawl2Answer is a complete RAG (Retrieval Augmented Generation) system that:

1. **Crawls** websites and extracts clean textual content
2. **Chunks** text into manageable segments
3. **Generates** embeddings for text chunks
4. **Stores** embeddings in a vector database
5. **Retrieves** relevant chunks for user queries
6. **Generates** answers strictly from retrieved content
7. **Exposes** functionality via a REST API

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Crawler   │───▶│  Text Extractor │───▶│   Text Chunker  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    REST API     │◀───│    Retriever    │◀───│    Embedder     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                               │                       │
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Vector Store   │◀───│  Vector Database│
                       └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
Crawl2Answer/
│
├── crawling/           # Website crawling functionality
│   └── crawler.py
│
├── extraction/         # HTML to text extraction
│   └── text_extractor.py
│
├── chunking/           # Text segmentation
│   └── chunker.py
│
├── embeddings/         # Vector embeddings generation
│   └── embedder.py
│
├── vector_store/       # Vector database operations
│   └── vector_db.py
│
├── retrieval/          # Content retrieval and ranking
│   └── retriever.py
│
├── api/                # REST API endpoints
│   └── main.py
│
├── config/             # Configuration management
│   └── settings.py
│
├── data/               # Data storage
│   ├── raw/           # Raw crawled content
│   ├── processed/     # Cleaned and processed text
│   └── embeddings/    # Vector database files
│
├── .env               # Environment variables
├── .env.example       # Environment variables template
├── .gitignore         # Git ignore rules
├── requirements.txt   # Python dependencies
├── run.sh            # Unix start script
├── run.bat           # Windows start script
└── README.md         # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation & Setup

#### Option 1: Automatic Setup (Recommended)

**Windows:**
```cmd
run.bat
```

**Linux/Mac:**
```bash
chmod +x run.sh
./run.sh
```

#### Option 2: Manual Setup

1. **Clone and navigate to the project**
   ```bash
   cd Crawl2Answer
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv crawl2answer_env
   ```

3. **Activate the virtual environment**
   
   **Windows:**
   ```cmd
   crawl2answer_env\Scripts\activate
   ```
   
   **Linux/Mac:**
   ```bash
   source crawl2answer_env/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env file with your configuration
   ```

6. **Start the API server**
   ```bash
   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## ⚙️ Configuration

Edit the `.env` file to configure the system:

```env
# Website to crawl
BASE_URL=https://docs.python.org/3/

# Embedding model (free option: sentence_transformers)
EMBEDDING_MODEL_TYPE=sentence_transformers
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2

# For OpenAI embeddings (requires API key)
# EMBEDDING_MODEL_TYPE=openai
# OPENAI_API_KEY=your_api_key_here

# Text processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# API settings
API_PORT=8000
```

## 🧪 Testing

After setting up the environment, you can test the various components:

### Option 1: Test Scripts
```bash
# Test crawler functionality
python tests/test_crawler_simple.py

# Test text extraction  
python tests/test_text_extraction.py

# Test text chunking
python tests/test_chunking_demo.py

# Test full pipeline
python tests/test_full_pipeline.py
```

### Option 2: API Testing
1. Start the API server:
   ```bash
   python -m uvicorn api.main:app --reload
   ```

2. Test individual endpoints:
   ```bash
   # Test crawler
   curl -X POST "http://localhost:8000/test-crawl" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://docs.python.org/3/tutorial/",
       "max_pages": 5,
       "max_depth": 2,
       "delay": 1.0
     }'
   
   # Test text extraction
   curl -X POST "http://localhost:8000/test-extraction" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://docs.python.org/3/tutorial/introduction.html",
       "delay": 1.0
     }'
   
   # Test text chunking
   curl -X POST "http://localhost:8000/test-chunking" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://docs.python.org/3/tutorial/introduction.html",
       "strategy": "smart",
       "delay": 1.0
     }'
```

### Features Tested

#### Crawler Features
- ✅ **Domain Restriction**: Only crawls internal links from the same domain
- ✅ **Smart Filtering**: Skips login pages, PDFs, APIs, and other non-content URLs  
- ✅ **Depth Control**: Limits crawling depth to prevent infinite loops
- ✅ **Rate Limiting**: Configurable delay between requests
- ✅ **Page Data**: Stores URL, title, HTML content, and metadata for each page
- ✅ **Link Extraction**: Finds and follows internal links automatically

#### Text Extraction Features
- ✅ **HTML Parsing**: Robust parsing with BeautifulSoup
- ✅ **Content Cleaning**: Removes navbars, footers, scripts, ads, cookie banners
- ✅ **Smart Detection**: Identifies main content areas automatically
- ✅ **Text Normalization**: Cleans whitespace, removes noise, filters quality
- ✅ **Rich Metadata**: Extracts titles, descriptions, headings, and statistics
- ✅ **Structured Output**: Type-safe data structures with comprehensive information

#### Text Chunking Features
- ✅ **Multiple Strategies**: Smart, fixed, sentence, and paragraph-based chunking
- ✅ **Boundary Detection**: Respects sentence and paragraph boundaries
- ✅ **Configurable Overlap**: Maintains context between chunks
- ✅ **Quality Filtering**: Removes low-quality chunks automatically
- ✅ **Statistical Analysis**: Provides comprehensive chunking metrics
- ✅ **Metadata Preservation**: Maintains source information and context

## 📖 Usage

### 1. Start the API Server

After running the setup script, the API will be available at:
- **API Endpoint:** http://localhost:8000
- **Documentation:** http://localhost:8000/docs
- **Alternative Documentation:** http://localhost:8000/redoc

### 2. Crawl a Website (Enhanced)

```bash
curl -X POST "http://localhost:8000/crawl" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://docs.python.org/3/tutorial/",
    "max_pages": 8,
    "max_depth": 2,
    "delay": 1.0
  }'
```

**Response includes:**
- List of crawled URLs with titles
- Domain information and statistics  
- Content size and crawling metadata
- Total pages processed and chunked

### 2a. Test Text Extraction (Single Page)

```bash
curl -X POST "http://localhost:8000/test-extraction" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://docs.python.org/3/tutorial/introduction.html",
    "delay": 1.0
  }'
```

**Features demonstrated:**
- HTML parsing and content extraction
- Removal of navigation, ads, and noise
- Main content area detection  
- Text cleaning and normalization
- Rich metadata extraction
- Content quality statistics

### 2b. Test Crawl (No Processing)

```bash
curl -X POST "http://localhost:8000/test-crawl" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://docs.python.org/3/tutorial/",
    "max_pages": 5,
    "max_depth": 2
  }'
```

This endpoint only crawls and returns URLs without processing content.

### 3. Ask Questions

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is a Python list?",
    "max_results": 3
  }'
```

### 4. Check System Status

```bash
curl http://localhost:8000/status
```

## 🔧 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | System information and stats |
| POST | `/crawl` | Crawl website and add to knowledge base (with enhanced text extraction) |
| POST | `/test-crawl` | Test crawl website and return URLs only (no content processing) |
| POST | `/test-extraction` | Test text extraction from a single page (with cleaning preview) |
| POST | `/ask` | Ask a question and get an answer |
| GET | `/status` | Get system status and statistics |
| DELETE | `/clear` | Clear the knowledge base |

## 🎛️ Advanced Configuration

### Embedding Models

**Option 1: SentenceTransformers (Free)**
```env
EMBEDDING_MODEL_TYPE=sentence_transformers
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
```

**Option 2: OpenAI (Requires API Key)**
```env
EMBEDDING_MODEL_TYPE=openai
EMBEDDING_MODEL_NAME=text-embedding-ada-002
OPENAI_API_KEY=your_api_key_here
```

### Text Chunking

Adjust chunk size and overlap for optimal performance:
```env
CHUNK_SIZE=1000        # Characters per chunk
CHUNK_OVERLAP=200      # Overlap between chunks
```

### Retrieval Settings

Configure retrieval behavior:
```env
RETRIEVAL_K=5              # Number of chunks to retrieve
SIMILARITY_THRESHOLD=0.1   # Minimum similarity score
```

## 🧪 Example Workflow

1. **Configure your target website** in `.env`
2. **Start the API server** using `run.bat` or `run.sh`
3. **Crawl the website:**
   ```json
   POST /crawl
   {
     "url": "https://docs.python.org/3/",
     "max_pages": 10
   }
   ```
4. **Ask questions:**
   ```json
   POST /ask
   {
     "question": "How do I create a Python function?",
     "max_results": 5
   }
   ```
5. **Get structured answers** with source references

## 🛠️ Development

### Project Components

- **Crawler:** Fetches web pages with rate limiting
- **Text Extractor:** Cleans HTML and extracts readable content
- **Chunker:** Splits text into overlapping segments
- **Embedder:** Generates vector representations using SentenceTransformers or OpenAI
- **Vector Store:** FAISS-based similarity search
- **Retriever:** Finds relevant content for queries
- **API:** FastAPI-based REST interface

### Adding New Features

1. Each component is modular and can be extended independently
2. Add new endpoints in `api/main.py`
3. Extend configuration in `config/settings.py`
4. Update requirements in `requirements.txt`

## 🐛 Troubleshooting

**Common Issues:**

1. **Import errors:** Make sure virtual environment is activated
2. **Port conflicts:** Change `API_PORT` in `.env`
3. **Memory issues:** Reduce `CHUNK_SIZE` or `MAX_PAGES`
4. **OpenAI errors:** Check your API key in `.env`

**Logs:**
- Check console output for detailed error messages
- Adjust `LOG_LEVEL` in `.env` for more/less verbose logging

## 📋 Requirements

- Python 3.8+
- 4GB+ RAM (for embedding models)
- Internet connection (for crawling and downloading models)
- ~500MB disk space (for models and data)

## 🏗️ Built With

- **FastAPI** - REST API framework
- **SentenceTransformers** - Embedding generation
- **FAISS** - Vector similarity search
- **BeautifulSoup** - HTML parsing
- **Requests** - HTTP client
- **Pydantic** - Data validation

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

For questions and support:
- Check the API documentation at `/docs`
- Review the configuration in `.env`
- Check the console logs for detailed error messages
