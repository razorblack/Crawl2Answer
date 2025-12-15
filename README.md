# Crawl2Answer

**Crawl. Retrieve. Answer.**

A comprehensive Q&A support bot using Retrieval Augmented Generation (RAG) that crawls websites, extracts clean textual content, and provides accurate answers based on retrieved information.

## 🎯 Project Overview

Crawl2Answer is a complete **Retrieval Augmented Generation (RAG) system** that transforms any website into an intelligent Q&A bot. The system follows a systematic 8-step implementation process:

**Core Pipeline:**
1. **Web Crawling** - Discovers and downloads web pages
2. **Text Extraction** - Extracts clean content from HTML
3. **Text Chunking** - Segments content into manageable pieces
4. **Embedding Generation** - Creates vector representations of text
5. **Vector Database** - Stores embeddings for similarity search
6. **Retrieval & Answer Generation** - Finds relevant content and generates answers
7. **REST API** - Exposes functionality via HTTP endpoints
8. **Documentation** - Comprehensive usage guide and examples

**Key Features:**
- ✅ **Intelligent Web Crawling** with depth and breadth control
- ✅ **Advanced Text Extraction** with noise removal and content cleaning
- ✅ **Semantic Chunking** for optimal context preservation  
- ✅ **Multiple Embedding Models** (SentenceTransformers, OpenAI)
- ✅ **High-Performance Vector Search** using FAISS
- ✅ **Context-Grounded Answers** with source attribution
- ✅ **Production-Ready REST API** with FastAPI
- ✅ **Comprehensive Testing Suite** for all components
- ✅ **Interactive Documentation** with Swagger UI

**Use Cases:**
- Documentation Q&A systems
- Customer support automation  
- Knowledge base search and retrieval
- Content-based chatbots
- Research and information discovery

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

## 🚀 Steps to Run the Crawler

### Quick Start (Recommended)

1. **Start the API Server**
   ```bash
   # Install dependencies (if not already done)
   pip install fastapi uvicorn requests beautifulsoup4 sentence-transformers faiss-cpu

   # Start the server
   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Crawl a Website**
   ```bash
   curl -X POST "http://localhost:8000/crawl" \
     -H "Content-Type: application/json" \
     -d '{
       "baseUrl": "https://docs.python.org/3/tutorial/",
       "max_pages": 10,
       "max_depth": 2,
       "delay": 1.0
     }'
   ```

3. **Wait for Processing**
   - The crawler will discover and download pages
   - Text extraction removes HTML and noise
   - Content is chunked into segments
   - Embeddings are generated and stored
   - Vector database is updated

4. **Test with Questions**
   ```bash
   curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "What are Python lists?"}'
   ```

### Step-by-Step Process

**Phase 1: Setup Environment**
```bash
# 1. Clone repository and create virtual environment
python -m venv crawl2answer_env
crawl2answer_env\Scripts\activate  # Windows
# source crawl2answer_env/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure settings (optional)
cp .env.example .env  # Edit .env if needed
```

**Phase 2: Start the System**
```bash
# 4. Launch the API server
python -m uvicorn api.main:app --port 8000 --reload

# Alternative: Use the startup utility (provides better logging)
python tests/start_api.py
```

**Phase 3: Run the Crawler**
```bash
# 5. Initiate crawling via API
curl -X POST "http://localhost:8000/crawl" \
  -H "Content-Type: application/json" \
  -d '{
    "baseUrl": "https://your-target-website.com",
    "max_pages": 20,
    "max_depth": 3,
    "delay": 1.5
  }'
```

**Expected Output:**
```json
{
  "status": "success",
  "message": "Crawling completed successfully",
  "base_url": "https://your-target-website.com", 
  "pages_crawled": 18,
  "chunks_created": 127,
  "embeddings_generated": 127,
  "database_updated": true,
  "processing_time": 45.7
}
```

### Alternative: Component Testing

For testing individual components without the API:

```bash
# Test Step 1: Web Crawling
python tests/test_crawler_simple.py

# Test Step 2: Text Extraction  
python tests/test_text_extraction.py

# Test Step 3: Text Chunking
python tests/test_chunking_demo.py

# Test Step 4: Embedding Generation
python tests/test_step5_embeddings.py

# Test Step 5: Vector Database
python tests/test_step5_embeddings.py

# Test Step 6: RAG Pipeline
python tests/test_step6_rag.py

# Test Step 7: API Endpoints
python tests/test_step7_api.py

# Direct API Testing
python tests/test_api_direct.py

# Full Pipeline Test
python tests/test_full_pipeline.py

# API Server Utilities
python tests/start_api.py              # Start API server with logging
python tests/step7_testing_guide.py    # Manual testing guide with curl commands

# Run All Tests (if pytest is installed)
pytest tests/ -v                       # Run all tests with verbose output
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
├── tests/              # Test files and utilities
│   ├── test_crawler_simple.py     # Basic crawler testing
│   ├── test_text_extraction.py    # Text extraction testing
│   ├── test_chunking_demo.py       # Text chunking testing
│   ├── test_step5_embeddings.py    # Embedding generation testing
│   ├── test_step6_rag.py           # RAG pipeline testing
│   ├── test_step7_api.py           # API endpoint testing
│   ├── test_api_direct.py          # Direct API testing
│   ├── test_full_pipeline.py       # Full pipeline integration
│   ├── start_api.py                # API server startup utility
│   ├── step7_testing_guide.py      # Manual testing guide
│   └── __init__.py                 # Python package marker
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

## ❓ How to Test the /ask Endpoint

### Basic Testing

Once you have crawled content and the vector database is populated, you can test the question-answering functionality:

**1. Simple Question**
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is artificial intelligence?"}'
```

**2. Specific Technical Question**  
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "How do neural networks process information?"}'
```

**3. Complex Query**
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the differences between machine learning and deep learning?"}'
```

### Response Format

The `/ask` endpoint returns structured JSON responses:

```json
{
  "question": "What is artificial intelligence?",
  "answer": "Based on the available context, Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence...",
  "sources": [
    "sample_ai_article.html",
    "introduction_to_ai.html"
  ],
  "confidence": 0.87,
  "retrieval_time": 0.045,
  "generation_time": 1.234
}
```

### Advanced Testing

**Test with Browser (Interactive Documentation)**
1. Navigate to: `http://localhost:8000/docs`
2. Find the `/ask` endpoint
3. Click "Try it out"
4. Enter your question in the request body
5. Click "Execute" to see the response

**Test with Python**
```python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={"question": "Your question here"}
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
print(f"Confidence: {result['confidence']}")
```

### Troubleshooting /ask Endpoint

**No Results Returned:**
- Ensure vector database has content (run `/crawl` first)
- Try questions related to your crawled content
- Check that embeddings were generated successfully

**Low Quality Answers:**
- Crawl more comprehensive content
- Try different question phrasing  
- Check if the question matches your content domain

**Slow Response Times:**
- Reduce the number of retrieved chunks
- Use smaller embedding models
- Optimize vector database settings

## 📝 Example Questions and Answers

### If you crawled AI/ML documentation:

**Q: "What is machine learning?"**
```json
{
  "answer": "Machine Learning (ML) is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. ML focuses on developing computer programs that can access data and use it to learn for themselves.",
  "sources": ["machine_learning_basics.html", "ai_overview.html"],
  "confidence": 0.92
}
```

**Q: "How do neural networks work?"**  
```json
{
  "answer": "Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes (neurons) that process information by passing signals through weighted connections. The network learns by adjusting these weights based on training data to minimize prediction errors.",
  "sources": ["neural_networks_guide.html", "deep_learning_intro.html"], 
  "confidence": 0.88
}
```

**Q: "What are the applications of computer vision?"**
```json
{
  "answer": "Computer vision has numerous applications including image recognition, object detection, facial recognition, medical image analysis, autonomous vehicles, augmented reality, quality control in manufacturing, and surveillance systems.",
  "sources": ["computer_vision_applications.html", "cv_use_cases.html"],
  "confidence": 0.85
}
```

### If you crawled Python documentation:

**Q: "How do I create a list in Python?"**
```json
{
  "answer": "In Python, you can create a list using square brackets []. For example: my_list = [1, 2, 3] creates a list with three elements. You can also create an empty list with empty_list = [] or use the list() constructor.",
  "sources": ["python_lists_tutorial.html", "data_structures.html"],
  "confidence": 0.94
}
```

**Q: "What is the difference between a list and a tuple?"**
```json
{
  "answer": "Lists are mutable (can be changed) and use square brackets [], while tuples are immutable (cannot be changed) and use parentheses (). Lists have methods like append() and remove(), while tuples have fewer methods due to their immutable nature.",
  "sources": ["python_data_types.html", "lists_vs_tuples.html"],
  "confidence": 0.90
}
```

### If you crawled a company website:

**Q: "What services does the company offer?"**
```json
{
  "answer": "Based on the website content, the company offers web development services, mobile app development, cloud infrastructure solutions, and digital consulting services for businesses of all sizes.",
  "sources": ["services.html", "about_us.html"],
  "confidence": 0.89
}
```
   
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

## � Limitations and Future Improvements

### Current Limitations

**1. Content Processing**
- Limited to HTML text extraction (no PDF, Word, or other formats)
- No support for JavaScript-heavy dynamic content
- Basic text cleaning may miss complex formatting
- No image or multimedia content analysis

**2. Crawling Constraints**
- Respects robots.txt but may still overwhelm small servers
- No distributed crawling for large-scale operations
- Limited error handling for network timeouts
- No incremental updates (full re-crawl required)

**3. Question Answering**
- Relies heavily on the quality of crawled content
- May struggle with questions requiring multi-hop reasoning
- No conversation memory or context persistence
- Limited factual accuracy validation

**4. Technical Limitations**
- Single-threaded processing (no parallel crawling)
- In-memory vector database (limited scalability)
- No user authentication or rate limiting
- Basic embedding model (sentence-transformers)

**5. Deployment Constraints**
- Local development setup only
- No production deployment configuration
- Limited monitoring and logging
- No backup or disaster recovery

### Future Improvements

#### Phase 1: Enhanced Content Processing
- [ ] **Multi-format Support**: Add PDF, Word, PowerPoint extraction
- [ ] **Dynamic Content**: Implement Selenium for JavaScript rendering
- [ ] **Media Processing**: Extract text from images using OCR
- [ ] **Better Cleaning**: Advanced text preprocessing and normalization

#### Phase 2: Advanced Crawling
- [ ] **Distributed Crawling**: Multi-worker parallel processing
- [ ] **Incremental Updates**: Smart re-crawling of changed content
- [ ] **Advanced Scheduling**: Configurable crawl frequencies
- [ ] **Site Maps**: XML sitemap parsing for comprehensive coverage

#### Phase 3: Improved Question Answering
- [ ] **Better LLMs**: Integration with GPT-4, Claude, or local models
- [ ] **Multi-hop Reasoning**: Complex query decomposition and synthesis
- [ ] **Fact Checking**: Source verification and confidence scoring
- [ ] **Conversation Context**: Multi-turn question answering

#### Phase 4: Production Features
- [ ] **User Authentication**: API key management and user sessions
- [ ] **Rate Limiting**: Request throttling and quota management
- [ ] **Monitoring**: Performance metrics and health checks
- [ ] **Caching**: Intelligent response caching for common queries

#### Phase 5: Scalability & Performance
- [ ] **Cloud Databases**: PostgreSQL with pgvector or Pinecone
- [ ] **Async Processing**: Full async/await implementation
- [ ] **Load Balancing**: Multi-instance deployment support
- [ ] **Edge Computing**: CDN integration for global performance

#### Phase 6: Advanced Analytics
- [ ] **Query Analytics**: Popular questions and usage patterns
- [ ] **Content Insights**: Most valuable sources and gaps
- [ ] **Performance Tracking**: Response time and accuracy metrics
- [ ] **A/B Testing**: Different retrieval and generation strategies

### Contributing Ideas

**Easy Wins** (Good for beginners):
- Add more file format support
- Improve error messages and logging
- Create additional test cases
- Enhance documentation with more examples

**Medium Complexity**:
- Implement conversation memory
- Add query expansion and refinement
- Create a web interface
- Optimize embedding and retrieval performance

**Advanced Projects**:
- Distributed crawling system
- Real-time content updates
- Advanced reasoning capabilities
- Production deployment automation

### Technology Roadmap

**Short Term (1-3 months)**:
- Multi-format content support
- Better error handling and logging
- Conversation context management
- Performance optimizations

**Medium Term (3-6 months)**:  
- Production deployment features
- Advanced crawling capabilities
- Enhanced question answering
- Monitoring and analytics

**Long Term (6+ months)**:
- Distributed architecture
- Advanced AI capabilities
- Enterprise features
- Cloud-native deployment

---

## 🎯 Getting Started Next Steps

1. **Quick Test**: Run the basic pipeline with a simple website
   ```bash
   python -m uvicorn api.main:app --reload
   # Then crawl a small site and ask questions
   ```

2. **Custom Domain**: Crawl content specific to your use case  
   ```bash
   curl -X POST "http://localhost:8000/crawl" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://your-domain.com", "max_pages": 10}'
   ```

3. **Experiment**: Try different questions and see what works well
   - Start with simple factual questions
   - Test domain-specific terminology
   - Observe answer quality and source attribution

4. **Contribute**: Pick a limitation above and help improve the system!
   - Check the contributing ideas for good starting points
   - Submit issues for bugs or feature requests
   - Create pull requests for improvements

5. **Deploy**: Consider production deployment for real-world usage
   - Review the future improvements roadmap
   - Implement authentication and rate limiting
   - Set up monitoring and logging

---

*This project demonstrates a complete RAG (Retrieval Augmented Generation) pipeline suitable for learning, prototyping, and small-scale production use. For enterprise applications, consider the advanced improvements listed above.*

---

## � Troubleshooting

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
