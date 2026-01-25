"""
Enhanced RAG System Usage Examples
===================================

This script demonstrates practical usage of the enhanced RAG system
with one-time CSV ingestion and agent-integrated query processing.
"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag.enhanced_rag_system import EnhancedRAGSystem


def main():
    """Demonstrate RAG system usage"""

    print("🚀 Enhanced RAG System Usage Examples")
    print("=" * 50)

    # Initialize data path
    data_path = project_root.parent / "QC Anonymized Study Files"

    if not data_path.exists():
        print(f"❌ Data directory not found: {data_path}")
        print("Please ensure the QC Anonymized Study Files directory exists.")
        return

    print(f"📁 Data source: {data_path}")

    # Initialize RAG system
    print("\n🔧 Initializing RAG System...")
    rag_system = EnhancedRAGSystem(data_path)

    try:
        # One-time data ingestion
        print("📥 Performing one-time data ingestion...")
        ingested = rag_system.initialize()

        if ingested:
            print("✅ Knowledge graph built from fresh data")
        else:
            print("✅ Knowledge graph loaded from cache")

        # Show system status
        status = rag_system.get_system_status()
        print("\n📊 System Status:")
        print(f"   • Nodes: {status['graph_statistics']['nodes']}")
        print(f"   • Edges: {status['graph_statistics']['edges']}")
        print(f"   • Node types: {list(status['graph_statistics']['node_types'].keys())}")

        # Example queries
        example_queries = [
            {
                'query': 'How many patients are enrolled across all studies?',
                'description': 'Basic factual query about patient enrollment'
            },
            {
                'query': 'What are the most common adverse events?',
                'description': 'Analytical query about safety data patterns'
            },
            {
                'query': 'Why might there be data quality issues in Study 5?',
                'description': 'Diagnostic query requiring root cause analysis'
            },
            {
                'query': 'What actions should we take to improve patient retention?',
                'description': 'Prescriptive query requiring agent recommendations'
            },
            {
                'query': 'Which studies are at risk of delays?',
                'description': 'Predictive query about potential issues'
            }
        ]

        print("\n🔍 Running Example Queries...")
        print("-" * 50)

        for i, example in enumerate(example_queries, 1):
            print(f"\n📋 Query {i}: {example['description']}")
            print(f"❓ {example['query']}")

            # Process query
            response = rag_system.query(example['query'])

            # Display results
            if response['success']:
                print("✅ Response generated successfully")

                # Show answer
                answer = response['answer']
                print(f"💬 Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")

                # Show agent insights if available
                if response.get('agent_insights'):
                    print(f"🤖 Agent Insights: {len(response['agent_insights'])} insights provided")

                if response.get('agent_recommendations'):
                    print(f"🎯 Recommendations: {len(response['agent_recommendations'])} actions suggested")

                # Show routing info
                routing = response.get('routing', {})
                print(f"🔀 Processing Strategy: {routing.get('strategy', 'unknown')}")
            else:
                print("❌ Query processing failed")
                if 'error' in response:
                    print(f"   Error: {response['error']}")

        # Advanced usage example
        print("\n🎯 Advanced Usage Example")
        print("-" * 30)

        # Query with context
        context = {
            'user_role': 'clinical_monitor',
            'study_focus': 'Study_1',
            'time_range': 'last_30_days'
        }

        advanced_query = "What are the critical issues requiring immediate attention?"
        print(f"❓ Advanced Query: {advanced_query}")
        print(f"📋 Context: {context}")

        response = rag_system.query(advanced_query, context)

        if response['success']:
            print("✅ Advanced query processed")
            print(f"💬 Response: {response['answer'][:300]}{'...' if len(response['answer']) > 300 else ''}")

            if response.get('agent_recommendations'):
                print("🎯 Key Recommendations:")
                for i, rec in enumerate(response['agent_recommendations'][:3], 1):
                    print(f"   {i}. {rec}")

        # Performance demonstration
        print("\n⚡ Performance Demonstration")
        print("-" * 30)

        import time

        test_queries = [
            "Patient count in Study 1?",
            "SAE summary across studies?",
            "Data quality metrics?",
            "Visit completion rates?"
        ]

        print("Running performance test with 4 queries...")
        start_time = time.time()

        for query in test_queries:
            rag_system.query(query)

        total_time = time.time() - start_time
        avg_time = total_time / len(test_queries)

        print(f"Total time: {total_time:.2f}s")
        print(f"Average per query: {avg_time:.2f}s")

        # System rebuild demonstration
        print("\n🔄 Knowledge Graph Management")
        print("-" * 30)

        print("💡 To rebuild the knowledge graph with fresh data:")
        print("   rag_system.rebuild_knowledge_graph()")
        print("   # This forces reingestion of all CSV files")

        print("\n📈 Usage Tips:")
        print("   • The system performs one-time ingestion automatically")
        print("   • Use context parameter for role-specific responses")
        print("   • Agent integration provides intelligent recommendations")
        print("   • Query routing optimizes processing based on complexity")

        print("\n✅ Enhanced RAG System demonstration completed!")
        return 0

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1


def interactive_mode():
    """Run in interactive mode for manual testing"""
    print("🔍 Interactive RAG System Testing")
    print("=" * 40)

    # Initialize system
    data_path = project_root.parent / "QC Anonymized Study Files"
    rag_system = EnhancedRAGSystem(data_path)

    print("📥 Initializing system...")
    rag_system.initialize()

    print("💬 Enter your queries (type 'quit' to exit):")

    while True:
        try:
            query = input("\n❓ Query: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                break

            if not query:
                continue

            print("🔄 Processing...")
            response = rag_system.query(query)

            if response['success']:
                print("✅ Answer:")
                print(response['answer'])

                if response.get('agent_recommendations'):
                    print("\n🎯 Recommendations:")
                    for rec in response['agent_recommendations'][:3]:
                        print(f"   • {rec}")

                routing = response.get('routing', {})
                print(f"Strategy: {routing.get('strategy', 'unknown')}")
            else:
                print("❌ Error:", response.get('error', 'Unknown error'))

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        exit_code = main()
        sys.exit(exit_code)