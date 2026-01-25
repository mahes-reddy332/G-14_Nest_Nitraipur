"""
Test and Demo Script for Enhanced RAG System
===========================================

This script demonstrates the enhanced RAG pipeline with:
1. One-time CSV ingestion into knowledge graph
2. Agent-integrated query processing
3. Intelligent query routing
4. Performance validation
"""

import sys
import os
import time
from pathlib import Path
import logging
import json
from typing import Dict, List, Any

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag.enhanced_rag_system import EnhancedRAGSystem, QueryType
from config.settings import DEFAULT_AGENT_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGSystemTester:
    """Test harness for the enhanced RAG system"""

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.rag_system = EnhancedRAGSystem(data_path)
        self.test_results = []

    def run_full_test_suite(self) -> Dict[str, Any]:
        """Run complete test suite"""
        logger.info("🚀 Starting Enhanced RAG System Test Suite")

        test_results = {
            'initialization': False,
            'ingestion': False,
            'queries': [],
            'performance': {},
            'errors': []
        }

        try:
            # Test 1: System Initialization
            logger.info("📋 Test 1: System Initialization")
            start_time = time.time()
            ingested = self.rag_system.initialize()
            init_time = time.time() - start_time

            test_results['initialization'] = True
            test_results['ingestion'] = ingested
            test_results['performance']['initialization_time'] = init_time

            # Store graph statistics
            graph_nodes = len(self.rag_system.kg_builder.knowledge_graph.graph.nodes())
            graph_edges = len(self.rag_system.kg_builder.knowledge_graph.graph.edges())
            test_results['performance']['graph_nodes'] = graph_nodes
            test_results['performance']['graph_edges'] = graph_edges

            logger.info(".2f")
            logger.info(f"   📊 Graph contains {graph_nodes} nodes")
            logger.info(f"   📊 Graph contains {graph_edges} edges")

            # Test 2: Query Processing Tests
            logger.info("📋 Test 2: Query Processing Tests")
            query_tests = self._run_query_tests()
            test_results['queries'] = query_tests

            # Test 3: Performance Tests
            logger.info("📋 Test 3: Performance Tests")
            performance_results = self._run_performance_tests()
            test_results['performance'].update(performance_results)

            # Test 4: System Status
            logger.info("📋 Test 4: System Status Check")
            status = self.rag_system.get_system_status()
            test_results['system_status'] = status

            logger.info("✅ All tests completed successfully!")

        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            test_results['errors'].append(str(e))

        return test_results

    def _run_query_tests(self) -> List[Dict[str, Any]]:
        """Run various query tests"""
        test_queries = [
            {
                'query': 'How many patients are enrolled in Study 1?',
                'expected_type': QueryType.FACTUAL,
                'description': 'Basic factual query'
            },
            {
                'query': 'What are the common adverse events across all studies?',
                'expected_type': QueryType.ANALYTICAL,
                'description': 'Analytical query requiring pattern analysis'
            },
            {
                'query': 'Why are there delays in patient visits for Study 5?',
                'expected_type': QueryType.DIAGNOSTIC,
                'description': 'Diagnostic query requiring root cause analysis'
            },
            {
                'query': 'What actions should we take to improve data quality?',
                'expected_type': QueryType.PRESCRIPTIVE,
                'description': 'Prescriptive query requiring agent recommendations'
            },
            {
                'query': 'Which patients are at risk of dropping out?',
                'expected_type': QueryType.PREDICTIVE,
                'description': 'Predictive query requiring risk assessment'
            }
        ]

        results = []

        for i, test_case in enumerate(test_queries, 1):
            logger.info(f"   🔍 Query Test {i}: {test_case['description']}")

            try:
                start_time = time.time()
                response = self.rag_system.query(test_case['query'])
                processing_time = time.time() - start_time

                # Validate response
                success = self._validate_query_response(response, test_case)

                result = {
                    'test_id': i,
                    'query': test_case['query'],
                    'description': test_case['description'],
                    'expected_type': test_case['expected_type'].value,
                    'success': success,
                    'processing_time': processing_time,
                    'response_length': len(response.get('answer', '')),
                    'has_agent_insights': bool(response.get('agent_insights')),
                    'has_recommendations': bool(response.get('agent_recommendations')),
                    'routing_strategy': response.get('routing', {}).get('strategy'),
                    'actual_query_type': response.get('metadata', {}).get('query_type')
                }

                results.append(result)

                status = "✅" if success else "⚠️"
                logger.info(f"      {status} {processing_time:.2f}s - {result['routing_strategy']}")

            except Exception as e:
                logger.error(f"      ❌ Query failed: {e}")
                results.append({
                    'test_id': i,
                    'query': test_case['query'],
                    'success': False,
                    'error': str(e)
                })

        return results

    def _validate_query_response(self, response: Dict[str, Any], test_case: Dict[str, Any]) -> bool:
        """Validate that a query response is reasonable"""
        if not response.get('success', False):
            return False

        # Check basic response structure
        if 'answer' not in response:
            return False

        answer = response['answer']
        if not isinstance(answer, str) or len(answer.strip()) < 10:
            return False

        # Check metadata
        metadata = response.get('metadata', {})
        if 'processing_time_ms' not in metadata:
            return False

        # Check routing information
        routing = response.get('routing', {})
        if 'strategy' not in routing:
            return False

        # For prescriptive queries, check for agent recommendations
        if test_case['expected_type'] == QueryType.PRESCRIPTIVE:
            if not response.get('agent_recommendations'):
                return False

        return True

    def _run_performance_tests(self) -> Dict[str, Any]:
        """Run performance benchmarking tests"""
        logger.info("   ⚡ Running performance benchmarks...")

        performance_results = {
            'concurrent_queries': [],
            'memory_usage': 'N/A',
            'cache_effectiveness': {}
        }

        # Test query throughput
        test_queries = [
            "How many patients in Study 1?",
            "What are the SAE rates?",
            "Show visit completion trends",
            "Identify data quality issues",
            "Recommend process improvements"
        ]

        # Run queries sequentially and measure performance
        times = []
        for query in test_queries:
            start_time = time.time()
            response = self.rag_system.query(query)
            elapsed = time.time() - start_time
            times.append(elapsed)

            if response.get('success'):
                performance_results['concurrent_queries'].append({
                    'query': query,
                    'time': elapsed,
                    'success': True
                })

        if times:
            performance_results['avg_query_time'] = sum(times) / len(times)
            performance_results['min_query_time'] = min(times)
            performance_results['max_query_time'] = max(times)

        logger.info(".2f")
        logger.info(".2f")
        logger.info(".2f")
        return performance_results

    def demonstrate_agent_integration(self) -> Dict[str, Any]:
        """Demonstrate agent integration capabilities"""
        logger.info("🤖 Demonstrating Agent Integration")

        demo_queries = [
            "What should we do about the high SAE rate in Study 3?",
            "How can we improve patient retention across studies?",
            "Why are coding queries taking so long to resolve?",
            "What actions are needed to ensure data quality standards?"
        ]

        results = []

        for query in demo_queries:
            logger.info(f"   🎯 Processing: {query}")

            response = self.rag_system.query(query)

            agent_insights = response.get('agent_insights', [])
            recommendations = response.get('agent_recommendations', [])

            result = {
                'query': query,
                'agent_insights_count': len(agent_insights),
                'recommendations_count': len(recommendations),
                'agents_consulted': response.get('metadata', {}).get('agents_consulted', []),
                'routing_strategy': response.get('routing', {}).get('strategy')
            }

            results.append(result)

            logger.info(f"      📊 Agents consulted: {len(result['agents_consulted'])}")
            logger.info(f"      💡 Insights generated: {result['agent_insights_count']}")
            logger.info(f"      🎯 Recommendations: {result['recommendations_count']}")

        return results


def main():
    """Main test execution"""
    print("🧪 Enhanced RAG System Test Suite")
    print("=" * 50)

    # Determine data path
    script_dir = Path(__file__).parent
    data_path = script_dir.parent.parent / "QC Anonymized Study Files"

    if not data_path.exists():
        print(f"❌ Data path not found: {data_path}")
        return 1

    print(f"📁 Using data path: {data_path}")

    # Initialize tester
    tester = RAGSystemTester(data_path)

    try:
        # Run full test suite
        results = tester.run_full_test_suite()

        # Demonstrate agent integration
        agent_demo = tester.demonstrate_agent_integration()

        # Generate test report
        generate_test_report(results, agent_demo)

        # Summary
        print("\n📊 Test Summary:")
        print(f"   ✅ Initialization: {'PASS' if results['initialization'] else 'FAIL'}")
        print(f"   📥 Data Ingestion: {'FRESH' if results['ingestion'] else 'CACHED'}")

        successful_queries = sum(1 for q in results['queries'] if q.get('success', False))
        total_queries = len(results['queries'])
        print(f"   🔍 Query Tests: {successful_queries}/{total_queries} PASSED")

        if results['performance'].get('avg_query_time'):
            print(".2f")
        print(f"   🤖 Agent Integration: {len(agent_demo)} demos completed")

        if results['errors']:
            print(f"   ❌ Errors: {len(results['errors'])}")
            for error in results['errors']:
                print(f"      - {error}")

        return 0 if successful_queries == total_queries and results['initialization'] else 1

    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        return 1


def generate_test_report(test_results: Dict[str, Any], agent_demo: List[Dict[str, Any]]):
    """Generate detailed test report"""
    report_path = Path(__file__).parent / "rag_test_report.json"

    report = {
        'timestamp': time.time(),
        'test_results': test_results,
        'agent_demonstration': agent_demo,
        'summary': {
            'total_queries': len(test_results['queries']),
            'successful_queries': sum(1 for q in test_results['queries'] if q.get('success', False)),
            'average_query_time': test_results['performance'].get('avg_query_time', 0),
            'total_agent_insights': sum(len(d.get('agent_insights', [])) for d in agent_demo),
            'total_recommendations': sum(d.get('recommendations_count', 0) for d in agent_demo)
        }
    }

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"📄 Detailed test report saved to: {report_path}")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)