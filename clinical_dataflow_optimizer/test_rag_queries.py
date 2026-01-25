import sys
sys.path.append('.')
from rag.enhanced_rag_system import EnhancedRAGPipeline, QueryRouter, KnowledgeGraphBuilder
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Test the RAG pipeline with the built knowledge graph
data_path = Path('../QC Anonymized Study Files')
kg_builder = KnowledgeGraphBuilder(data_path)
pipeline = EnhancedRAGPipeline(kg_builder)

print('Testing Enhanced RAG Pipeline...')

# Test different types of queries
test_queries = [
    'How many patients are in Study 10?',
    'What are the most common coding issues across all studies?',
    'Show me patients with SAE events in Study 1',
    'What is the average visit completion rate?',
    'Identify patients with multiple coding issues'
]

for i, query in enumerate(test_queries, 1):
    print(f'\n--- Query {i}: {query} ---')
    try:
        # Route the query using the pipeline
        router = QueryRouter(pipeline)
        result = router.route_query(query)

        print(f'Query type: {result.get("query_type", "unknown")}')
        print(f'Strategy: {result.get("routing", {}).get("strategy", "unknown")}')
        print(f'Processing time: {result.get("routing", {}).get("processing_time_ms", 0):.2f}ms')

        # Show the answer
        answer = result.get('answer', 'No answer provided')
        print(f'Response: {answer[:300]}...' if len(answer) > 300 else f'Response: {answer}')

    except Exception as e:
        print(f'Error processing query: {e}')
        import traceback
        traceback.print_exc()