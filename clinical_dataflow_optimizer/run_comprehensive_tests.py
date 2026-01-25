import sys
sys.path.append('.')
from rag.test_enhanced_rag import RAGSystemTester
from pathlib import Path

# Run comprehensive tests
data_path = Path('../QC Anonymized Study Files')
tester = RAGSystemTester(data_path)

print('Running comprehensive RAG system tests...')
try:
    results = tester.run_full_test_suite()
    print('\nTest Results Summary:')
    print(f'Results keys: {list(results.keys())}')
    
    # Check if tests passed
    if 'success' in results and results['success']:
        print('✅ All tests completed successfully!')
    else:
        print('❌ Some tests failed')
        
    # Show performance metrics if available
    if 'performance' in results:
        perf = results['performance']
        print(f'\\nPerformance Metrics:')
        print(f'  Average query time: {perf.get("avg_query_time", 0):.3f}s')
        print(f'  Ingestion time: {perf.get("ingestion_time", 0):.1f}s')
        print(f'  Graph nodes: {perf.get("graph_nodes", 0)}')
        print(f'  Graph edges: {perf.get("graph_edges", 0)}')
except Exception as e:
    print(f'Test execution failed: {e}')
    import traceback
    traceback.print_exc()