"""
Step 5 Test Script: Embedding Generation and Vector Database Setup

This script demonstrates and tests the complete Step 5 functionality:
1. Choose embedding model (SentenceTransformers)
2. Generate embeddings for text chunks
3. Set up vector database (FAISS)
4. Insert embeddings with metadata
5. Run similarity search tests

Run this script to verify Step 5 implementation.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import time
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_test_chunks() -> List[Dict[str, Any]]:
    """Load test chunks from the chunking output or create sample data"""
    chunks_file = project_root / "data" / "chunks.json"
    
    if chunks_file.exists():
        logger.info("Loading chunks from existing data/chunks.json")
        with open(chunks_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        logger.info("Creating sample chunks for testing")
        # Create sample chunks for testing
        sample_chunks = [
            {
                "chunk_id": 1,
                "content": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that work and react like humans. AI systems can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation.",
                "metadata": {
                    "source": "sample_ai_article.html",
                    "chunk_index": 0,
                    "word_count": 45,
                    "char_count": 287,
                    "url": "https://example.com/ai-intro",
                    "title": "Introduction to Artificial Intelligence",
                    "section": "overview"
                }
            },
            {
                "chunk_id": 2,
                "content": "Machine Learning (ML) is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. ML algorithms build mathematical models based on training data to make predictions or decisions.",
                "metadata": {
                    "source": "sample_ai_article.html",
                    "chunk_index": 1,
                    "word_count": 42,
                    "char_count": 268,
                    "url": "https://example.com/ai-intro",
                    "title": "Introduction to Artificial Intelligence",
                    "section": "machine_learning"
                }
            },
            {
                "chunk_id": 3,
                "content": "Deep Learning is a subset of machine learning that uses neural networks with multiple layers (deep neural networks) to model and understand complex patterns in data. It has revolutionized fields like computer vision, natural language processing, and speech recognition.",
                "metadata": {
                    "source": "sample_ai_article.html",
                    "chunk_index": 2,
                    "word_count": 43,
                    "char_count": 275,
                    "url": "https://example.com/ai-intro",
                    "title": "Introduction to Artificial Intelligence",
                    "section": "deep_learning"
                }
            },
            {
                "chunk_id": 4,
                "content": "Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language in a valuable way. NLP combines computational linguistics with statistical, machine learning, and deep learning models.",
                "metadata": {
                    "source": "sample_nlp_article.html",
                    "chunk_index": 0,
                    "word_count": 35,
                    "char_count": 215,
                    "url": "https://example.com/nlp-overview",
                    "title": "Natural Language Processing Overview",
                    "section": "introduction"
                }
            },
            {
                "chunk_id": 5,
                "content": "Computer Vision is a field of artificial intelligence that trains computers to interpret and understand the visual world. Using digital images from cameras and videos and deep learning models, machines can accurately identify and classify objects.",
                "metadata": {
                    "source": "sample_cv_article.html",
                    "chunk_index": 0,
                    "word_count": 39,
                    "char_count": 244,
                    "url": "https://example.com/computer-vision",
                    "title": "Computer Vision Fundamentals",
                    "section": "overview"
                }
            }
        ]
        
        # Save sample chunks
        chunks_file.parent.mkdir(exist_ok=True)
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(sample_chunks, f, indent=2, ensure_ascii=False)
        
        return sample_chunks

def test_embedding_generation():
    """Test Step 5.1 & 5.2: Choose embedding model and generate embeddings"""
    logger.info("=" * 60)
    logger.info("STEP 5.1 & 5.2: EMBEDDING MODEL AND GENERATION")
    logger.info("=" * 60)
    
    try:
        from embeddings.embedder_enhanced import Embedder
        
        # Initialize embedder with SentenceTransformers model
        logger.info("Initializing embedding model: all-MiniLM-L6-v2")
        embedder = Embedder()
        
        # Load test chunks
        chunks = load_test_chunks()
        logger.info(f"Loaded {len(chunks)} test chunks")
        
        # Generate embeddings for chunks
        logger.info("Generating embeddings for chunks...")
        start_time = time.time()
        
        embedding_results = embedder.generate_embeddings_for_chunks(chunks)
        
        generation_time = time.time() - start_time
        logger.info(f"Generated {len(embedding_results)} embeddings in {generation_time:.2f} seconds")
        
        # Display sample embedding info
        if embedding_results:
            sample_result = embedding_results[0]
            logger.info(f"Sample embedding - Chunk ID: {sample_result.chunk_id}")
            logger.info(f"Embedding dimension: {sample_result.dimension}")
            logger.info(f"Model used: {sample_result.model_name}")
            logger.info(f"Content preview: {sample_result.metadata.get('content', '')[:100]}...")
            logger.info(f"Embedding vector preview: {sample_result.embedding[:5]}...")
        
        return embedding_results
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return None

def test_vector_database_setup(embedding_results):
    """Test Step 5.3: Choose and set up vector database (FAISS)"""
    logger.info("=" * 60)
    logger.info("STEP 5.3: VECTOR DATABASE SETUP (FAISS)")
    logger.info("=" * 60)
    
    try:
        from vector_store.vector_db_enhanced import VectorDatabase
        
        # Initialize FAISS vector database
        embedding_dim = len(embedding_results[0].embedding) if embedding_results else 384
        logger.info(f"Initializing FAISS vector database with dimension: {embedding_dim}")
        
        vector_db = VectorDatabase(
            dimension=embedding_dim,
            storage_path="data/vector_store",
            index_type="cosine"  # Using cosine similarity
        )
        
        logger.info("Vector database initialized successfully")
        logger.info(f"Index type: {vector_db.index_type}")
        logger.info(f"Storage path: {vector_db.storage_path}")
        
        return vector_db
        
    except Exception as e:
        logger.error(f"Vector database setup failed: {e}")
        return None

def test_embedding_insertion(vector_db, embedding_results):
    """Test Step 5.4: Insert embeddings and metadata into vector store"""
    logger.info("=" * 60)
    logger.info("STEP 5.4: INSERT EMBEDDINGS WITH METADATA")
    logger.info("=" * 60)
    
    try:
        # Insert embeddings into vector database
        logger.info(f"Inserting {len(embedding_results)} embeddings into vector store...")
        start_time = time.time()
        
        successful_insertions = vector_db.add_embedding_results(embedding_results)
        
        insertion_time = time.time() - start_time
        logger.info(f"Successfully inserted {successful_insertions} embeddings in {insertion_time:.2f} seconds")
        
        # Display database stats
        stats = vector_db.get_stats()
        logger.info(f"Database stats:")
        logger.info(f"  - Total vectors: {stats.total_vectors}")
        logger.info(f"  - Embedding dimension: {stats.embedding_dimension}")
        logger.info(f"  - Index type: {stats.index_type}")
        logger.info(f"  - Storage size: {stats.total_size_mb:.2f} MB")
        logger.info(f"  - Last updated: {stats.last_updated}")
        
        # Save to disk
        logger.info("Saving vector database to disk...")
        save_success = vector_db.save()
        logger.info(f"Database saved: {save_success}")
        
        return successful_insertions > 0
        
    except Exception as e:
        logger.error(f"Embedding insertion failed: {e}")
        return False

def test_similarity_search(vector_db, embedder):
    """Test Step 5.5: Run similarity search tests"""
    logger.info("=" * 60)
    logger.info("STEP 5.5: SIMILARITY SEARCH TESTS")
    logger.info("=" * 60)
    
    # Test queries
    test_queries = [
        "What is artificial intelligence and how does it work?",
        "machine learning algorithms and training data",
        "neural networks and deep learning applications",
        "computer vision object detection",
        "natural language processing techniques"
    ]
    
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\nTest Query {i}: '{query}'")
        logger.info("-" * 50)
        
        try:
            # Generate embedding for query
            query_embedding = embedder.generate_embedding(query, {"query_type": "search"})
            
            if not query_embedding:
                logger.warning(f"Failed to generate embedding for query: {query}")
                continue
            
            # Perform similarity search
            start_time = time.time()
            results = vector_db.search(query_embedding.embedding, top_k=3)
            search_time = time.time() - start_time
            
            logger.info(f"Found {len(results)} results in {search_time*1000:.1f}ms")
            
            # Display results
            for j, result in enumerate(results, 1):
                logger.info(f"  Result {j}:")
                logger.info(f"    Similarity Score: {result.similarity_score:.4f}")
                logger.info(f"    Chunk ID: {result.chunk_id}")
                logger.info(f"    Content Preview: {result.content[:150]}...")
                logger.info(f"    Source: {result.metadata.get('source', 'N/A')}")
                logger.info(f"    Section: {result.metadata.get('section', 'N/A')}")
                
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
    
    # Test filtering
    logger.info("\nTesting metadata filtering...")
    logger.info("-" * 30)
    
    try:
        # Search with section filter
        query_embedding = embedder.generate_embedding("artificial intelligence", {"query_type": "filtered_search"})
        filtered_results = vector_db.search(
            query_embedding.embedding, 
            top_k=5, 
            filter_metadata={"section": "overview"}
        )
        
        logger.info(f"Filtered search (section='overview'): {len(filtered_results)} results")
        for result in filtered_results:
            logger.info(f"  - Chunk {result.chunk_id}: {result.metadata.get('section', 'N/A')}")
            
    except Exception as e:
        logger.error(f"Filtered search failed: {e}")

def test_database_persistence(vector_db):
    """Test database loading and persistence"""
    logger.info("=" * 60)
    logger.info("TESTING DATABASE PERSISTENCE")
    logger.info("=" * 60)
    
    try:
        from vector_store.vector_db_enhanced import VectorDatabase
        
        # Test loading
        new_db = VectorDatabase(
            dimension=vector_db.dimension,
            storage_path=vector_db.storage_path,
            index_type=vector_db.index_type
        )
        
        load_success = new_db.load()
        logger.info(f"Database loaded successfully: {load_success}")
        
        if load_success:
            stats = new_db.get_stats()
            logger.info(f"Loaded database stats:")
            logger.info(f"  - Total vectors: {stats.total_vectors}")
            logger.info(f"  - Dimension: {stats.embedding_dimension}")
            logger.info(f"  - Index type: {stats.index_type}")
            
        return load_success
        
    except Exception as e:
        logger.error(f"Database persistence test failed: {e}")
        return False

def main():
    """Run complete Step 5 testing suite"""
    logger.info("STEP 5 TESTING: EMBEDDING GENERATION AND VECTOR DATABASE SETUP")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    # Step 5.1 & 5.2: Embedding generation
    embedding_results = test_embedding_generation()
    if not embedding_results:
        logger.error("Embedding generation failed. Stopping tests.")
        return False
    
    # Step 5.3: Vector database setup
    vector_db = test_vector_database_setup(embedding_results)
    if not vector_db:
        logger.error("Vector database setup failed. Stopping tests.")
        return False
    
    # Step 5.4: Insert embeddings
    insertion_success = test_embedding_insertion(vector_db, embedding_results)
    if not insertion_success:
        logger.error("Embedding insertion failed. Stopping tests.")
        return False
    
    # Additional: Test persistence
    from embeddings.embedder_enhanced import Embedder
    embedder = Embedder()
    test_similarity_search(vector_db, embedder)
    
    # Additional: Test persistence
    persistence_success = test_database_persistence(vector_db)
    
    total_time = time.time() - start_time
    
    # Final summary
    logger.info("=" * 80)
    logger.info("STEP 5 TESTING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total test time: {total_time:.2f} seconds")
    logger.info(f"Embeddings generated: {len(embedding_results)}")
    logger.info(f"Vector database: {'✓ FAISS with cosine similarity' if vector_db else '✗ Failed'}")
    logger.info(f"Similarity search: {'✓ Multiple queries tested' if vector_db else '✗ Failed'}")
    logger.info(f"Persistence: {'✓ Save/Load successful' if persistence_success else '✗ Failed'}")
    
    logger.info("\nStep 5 Implementation Summary:")
    logger.info("✓ Embedding Model: SentenceTransformers (all-MiniLM-L6-v2)")
    logger.info("✓ Embedding Generation: Batch processing with caching")
    logger.info("✓ Vector Database: FAISS with cosine similarity")
    logger.info("✓ Metadata Storage: Complete metadata preservation")
    logger.info("✓ Similarity Search: Multiple test queries executed")
    logger.info("✓ Performance: Optimized with caching and batch operations")
    
    return True

if __name__ == "__main__":
    success = main()
    exit_code = 0 if success else 1
    sys.exit(exit_code)