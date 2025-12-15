"""
Step 6 Test Script: Retrieval and Answer Generation

This script demonstrates and tests the complete Step 6 functionality:
1. Build a retrieval function - embed the user query and fetch top relevant chunks
2. Build an answer generation function - prepare prompts and call language model
3. Test with basic questions related to the crawled/sample website content

Run this script to verify Step 6 implementation.
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

def ensure_test_data():
    """Ensure we have test data from Step 5"""
    # Check if we have vector database from Step 5
    vector_db_path = project_root / "data" / "vector_store"
    if not vector_db_path.exists() or not (vector_db_path / "faiss_index.bin").exists():
        logger.warning("No vector database found. Running Step 5 test to create test data...")
        
        # Import and run Step 5 test
        try:
            sys.path.insert(0, str(project_root))
            from test_step5_embeddings import main as step5_main
            logger.info("Running Step 5 to generate test data...")
            step5_success = step5_main()
            if not step5_success:
                logger.error("Step 5 test failed. Cannot proceed with Step 6.")
                return False
        except Exception as e:
            logger.error(f"Failed to run Step 5 test: {e}")
            return False
    
    return True

def test_retrieval_function():
    """Test Step 6.1: Build retrieval function"""
    logger.info("=" * 60)
    logger.info("STEP 6.1: RETRIEVAL FUNCTION TEST")
    logger.info("=" * 60)
    
    try:
        from retrieval.retriever_enhanced import DocumentRetriever
        from config.settings import Settings
        
        # Initialize retriever
        settings = Settings()
        logger.info(f"Using vector database path: {settings.VECTOR_DB_PATH}")
        retriever = DocumentRetriever(settings)
        
        # Check database manually
        db_info = retriever.get_database_info()
        logger.info(f"Database info: {db_info}")
        
        # Test health check
        health = retriever.health_check()
        logger.info(f"Retriever health check: {health}")
        
        if not health.get("vector_db_ready", False) and health.get("total_documents", 0) == 0:
            logger.error("No vector database found. Running Step 5 first...")
            # Import and run Step 5 test
            from test_step5_embeddings import main as step5_main
            step5_success = step5_main()
            if step5_success:
                # Reinitialize retriever
                retriever = DocumentRetriever(settings)
                health = retriever.health_check()
                logger.info(f"Retriever health check after Step 5: {health}")
        
        if not health.get("embedder_ready", False):
            logger.error("Embedder not ready")
            return None
        
        # Test queries for retrieval
        test_queries = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "Explain deep learning and neural networks",
            "What is computer vision used for?",
            "Natural language processing techniques"
        ]
        
        retrieval_results = []
        
        for query in test_queries:
            logger.info(f"\nTesting query: '{query}'")
            logger.info("-" * 40)
            
            # Test retrieval
            start_time = time.time()
            result = retriever.retrieve(query, top_k=3, similarity_threshold=0.2)
            retrieval_time = time.time() - start_time
            
            logger.info(f"Retrieved {result.total_chunks} chunks in {retrieval_time:.3f}s")
            
            if result.total_chunks > 0:
                logger.info(f"Top similarity score: {result.relevance_scores[0]:.4f}")
                logger.info(f"Sources found: {len(result.get_unique_sources())}")
                
                # Show top result
                top_chunk = result.chunks[0]
                logger.info(f"Top result content: {top_chunk['content'][:150]}...")
                logger.info(f"Source: {top_chunk['metadata'].get('source', 'N/A')}")
            else:
                logger.warning("No results found for this query")
            
            retrieval_results.append(result)
        
        # Test retrieval stats
        stats = retriever.get_retrieval_stats()
        logger.info(f"\nRetrieval Statistics:")
        logger.info(f"Total queries: {stats['total_queries']}")
        logger.info(f"Average retrieval time: {stats['average_retrieval_time']:.3f}s")
        
        db_info = stats['database_info']
        logger.info(f"Database: {db_info['total_documents']} documents, {db_info['embedding_dimension']}D")
        
        return retriever, retrieval_results
        
    except Exception as e:
        logger.error(f"Retrieval function test failed: {e}")
        return None

def test_answer_generation_function():
    """Test Step 6.2: Build answer generation function"""
    logger.info("=" * 60)
    logger.info("STEP 6.2: ANSWER GENERATION FUNCTION TEST")
    logger.info("=" * 60)
    
    try:
        from generation.answer_generator import AnswerGenerator
        from retrieval.retriever_enhanced import DocumentRetriever
        from config.settings import Settings
        
        # Initialize components
        settings = Settings()
        retriever = DocumentRetriever(settings)
        answer_generator = AnswerGenerator(retriever, settings)
        
        # Test health check
        health = answer_generator.health_check()
        logger.info(f"Answer generator health check: {health}")
        
        # Test questions for answer generation
        test_questions = [
            "What is artificial intelligence and how does it work?",
            "What are the main applications of machine learning?",
            "How do neural networks process information?",
            "What problems can computer vision solve?",
            "How is natural language processing used in applications?"
        ]
        
        answer_results = []
        
        for question in test_questions:
            logger.info(f"\nGenerating answer for: '{question}'")
            logger.info("-" * 50)
            
            # Generate answer
            start_time = time.time()
            result = answer_generator.generate_answer(
                question=question,
                max_chunks=3,
                include_sources=True
            )
            generation_time = time.time() - start_time
            
            logger.info(f"Generated answer in {generation_time:.3f}s")
            logger.info(f"Confidence score: {result.confidence_score:.3f}")
            logger.info(f"Used {result.chunk_count} chunks from context")
            logger.info(f"Model: {result.model_used}")
            
            # Display answer
            logger.info(f"\nAnswer: {result.answer[:200]}...")
            
            # Show sources
            if result.sources:
                logger.info(f"Sources ({len(result.sources)}):")
                for i, source in enumerate(result.sources[:3], 1):
                    logger.info(f"  {i}. {source}")
            
            # Show retrieval scores
            if result.retrieval_scores:
                avg_score = sum(result.retrieval_scores) / len(result.retrieval_scores)
                logger.info(f"Average retrieval relevance: {avg_score:.4f}")
            
            answer_results.append(result)
        
        return answer_generator, answer_results
        
    except Exception as e:
        logger.error(f"Answer generation function test failed: {e}")
        return None

def test_complete_rag_pipeline():
    """Test Step 6.3: Complete RAG pipeline with basic questions"""
    logger.info("=" * 60)
    logger.info("STEP 6.3: COMPLETE RAG PIPELINE TEST")
    logger.info("=" * 60)
    
    try:
        from generation.answer_generator import AnswerGenerator
        from retrieval.retriever_enhanced import DocumentRetriever
        from config.settings import Settings
        
        # Initialize complete pipeline
        settings = Settings()
        retriever = DocumentRetriever(settings)
        answer_generator = AnswerGenerator(retriever, settings)
        
        # Comprehensive test questions related to our sample content
        comprehensive_questions = [
            {
                "question": "What is the definition of artificial intelligence?",
                "expected_topics": ["computer science", "intelligent machines", "human intelligence"]
            },
            {
                "question": "How does machine learning improve from experience?",
                "expected_topics": ["training data", "algorithms", "predictions"]
            },
            {
                "question": "What are neural networks and how do they work?",
                "expected_topics": ["layers", "deep learning", "patterns"]
            },
            {
                "question": "What can computer vision systems do?",
                "expected_topics": ["visual world", "images", "object detection"]
            },
            {
                "question": "How does natural language processing work?",
                "expected_topics": ["human language", "computational linguistics", "machine learning"]
            }
        ]
        
        successful_answers = 0
        total_questions = len(comprehensive_questions)
        
        for i, test_case in enumerate(comprehensive_questions, 1):
            question = test_case["question"]
            expected_topics = test_case["expected_topics"]
            
            logger.info(f"\n[Question {i}/{total_questions}] {question}")
            logger.info("=" * 70)
            
            # Generate complete answer
            result = answer_generator.generate_answer(
                question=question,
                max_chunks=5,
                include_sources=True
            )
            
            # Evaluate answer quality
            answer_lower = result.answer.lower()
            topic_matches = sum(1 for topic in expected_topics if topic.lower() in answer_lower)
            topic_coverage = topic_matches / len(expected_topics)
            
            # Display results
            logger.info(f"Generated Answer:")
            logger.info(f"{result.answer}")
            
            logger.info(f"\nAnswer Evaluation:")
            logger.info(f"  Confidence Score: {result.confidence_score:.3f}")
            logger.info(f"  Topic Coverage: {topic_coverage:.2f} ({topic_matches}/{len(expected_topics)} topics)")
            logger.info(f"  Generation Time: {result.generation_time:.3f}s")
            logger.info(f"  Context Chunks Used: {result.chunk_count}")
            
            if result.sources:
                logger.info(f"  Sources Referenced: {len(result.sources)}")
                for j, source in enumerate(result.sources, 1):
                    logger.info(f"    {j}. {source}")
            
            # Success criteria
            success = (
                result.confidence_score > 0.3 and
                topic_coverage > 0.5 and
                len(result.answer.split()) > 20 and
                result.chunk_count > 0
            )
            
            if success:
                successful_answers += 1
                logger.info("✓ PASS - Answer meets quality criteria")
            else:
                logger.info("✗ FAIL - Answer below quality threshold")
            
            logger.info("-" * 70)
        
        # Pipeline summary
        success_rate = successful_answers / total_questions
        
        # Get overall statistics
        gen_stats = answer_generator.get_generation_stats()
        
        logger.info(f"\nRAG PIPELINE SUMMARY:")
        logger.info(f"Questions processed: {total_questions}")
        logger.info(f"Successful answers: {successful_answers}")
        logger.info(f"Success rate: {success_rate:.2%}")
        
        logger.info(f"\nGeneration Statistics:")
        logger.info(f"Total questions: {gen_stats['generation']['total_questions']}")
        logger.info(f"Average generation time: {gen_stats['generation']['average_generation_time']:.3f}s")
        logger.info(f"Model used: {gen_stats['model']['name']}")
        
        logger.info(f"\nRetrieval Statistics:")
        logger.info(f"Total queries: {gen_stats['retrieval']['total_queries']}")
        logger.info(f"Average retrieval time: {gen_stats['retrieval']['average_retrieval_time']:.3f}s")
        
        return success_rate >= 0.6  # 60% success rate threshold
        
    except Exception as e:
        logger.error(f"Complete RAG pipeline test failed: {e}")
        return False

def demonstrate_advanced_features():
    """Demonstrate advanced features of the RAG system"""
    logger.info("=" * 60)
    logger.info("STEP 6.4: ADVANCED FEATURES DEMONSTRATION")
    logger.info("=" * 60)
    
    try:
        from generation.answer_generator import AnswerGenerator
        from retrieval.retriever_enhanced import DocumentRetriever
        from config.settings import Settings
        
        settings = Settings()
        retriever = DocumentRetriever(settings)
        answer_generator = AnswerGenerator(retriever, settings)
        
        # 1. Source-filtered retrieval
        logger.info("1. Source-filtered retrieval:")
        source_result = retriever.search_by_source(
            source_filter="sample_ai_article.html",
            query="artificial intelligence applications",
            top_k=3
        )
        logger.info(f"Found {source_result.total_chunks} chunks from specific source")
        
        # 2. Low confidence handling
        logger.info("\n2. Low confidence question handling:")
        result = answer_generator.generate_answer(
            "What is quantum computing in blockchain applications?",  # Likely not in our data
            max_chunks=3
        )
        logger.info(f"Low relevance query confidence: {result.confidence_score:.3f}")
        logger.info(f"Answer: {result.answer[:150]}...")
        
        # 3. Formatted answer with sources
        logger.info("\n3. Formatted answer with sources:")
        result = answer_generator.generate_answer("What is machine learning?")
        formatted = result.get_formatted_answer()
        logger.info(f"Formatted response:\n{formatted[:300]}...")
        
        # 4. Performance statistics
        logger.info("\n4. System performance:")
        stats = answer_generator.get_generation_stats()
        logger.info(f"Total processing time: {stats['generation']['total_generation_time']:.3f}s")
        logger.info(f"Database size: {stats['retrieval']['database_info']['total_documents']} documents")
        logger.info(f"Storage size: {stats['retrieval']['database_info']['storage_size_mb']:.2f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"Advanced features demonstration failed: {e}")
        return False

def main():
    """Run complete Step 6 testing suite"""
    logger.info("STEP 6 TESTING: RETRIEVAL AND ANSWER GENERATION")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    # Ensure test data exists
    if not ensure_test_data():
        logger.error("Failed to ensure test data. Stopping tests.")
        return False
    
    # Step 6.1: Test retrieval function
    retrieval_result = test_retrieval_function()
    if not retrieval_result:
        logger.error("Retrieval function test failed. Stopping tests.")
        return False
    
    retriever, retrieval_results = retrieval_result
    
    # Step 6.2: Test answer generation function
    generation_result = test_answer_generation_function()
    if not generation_result:
        logger.error("Answer generation function test failed. Stopping tests.")
        return False
    
    answer_generator, answer_results = generation_result
    
    # Step 6.3: Test complete RAG pipeline
    pipeline_success = test_complete_rag_pipeline()
    
    # Step 6.4: Demonstrate advanced features
    advanced_success = demonstrate_advanced_features()
    
    total_time = time.time() - start_time
    
    # Final summary
    logger.info("=" * 80)
    logger.info("STEP 6 TESTING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total test time: {total_time:.2f} seconds")
    
    # Component status
    logger.info(f"\nComponent Status:")
    logger.info(f"✓ Retrieval Function: {'PASS' if retrieval_result else 'FAIL'}")
    logger.info(f"✓ Answer Generation: {'PASS' if generation_result else 'FAIL'}")
    logger.info(f"✓ Complete RAG Pipeline: {'PASS' if pipeline_success else 'FAIL'}")
    logger.info(f"✓ Advanced Features: {'PASS' if advanced_success else 'FAIL'}")
    
    logger.info(f"\nStep 6 Implementation Summary:")
    logger.info(f"✓ Query Embedding: Generate embeddings for user questions")
    logger.info(f"✓ Similarity Search: Fetch top relevant document chunks")
    logger.info(f"✓ Context Preparation: Format retrieved context for LLM")
    logger.info(f"✓ Answer Generation: Use language model with context grounding")
    logger.info(f"✓ Source Attribution: Track and return source URLs")
    logger.info(f"✓ Quality Control: Confidence scoring and answer validation")
    
    overall_success = all([
        retrieval_result is not None,
        generation_result is not None,
        pipeline_success,
        advanced_success
    ])
    
    logger.info(f"\nOverall Status: {'✓ SUCCESS' if overall_success else '✗ FAILED'}")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    exit_code = 0 if success else 1
    sys.exit(exit_code)