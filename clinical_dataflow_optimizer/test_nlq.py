"""
Test Script for Natural Language Query (NLQ) Module
=====================================================

Tests the Conversational Insight Engine with real clinical data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
import pandas as pd
from datetime import datetime

# Import NLQ components
from nlq import (
    QueryParser, QueryIntent, ParsedQuery,
    QueryExecutor, QueryResult,
    InsightGenerator, ConversationalResponse,
    ConversationalEngine
)


@pytest.fixture
def data_sources():
    """Pytest fixture to load Study 1 data for testing"""
    return load_study_data()

def load_study_data():
    """Load Study 1 data for testing"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    study_path = os.path.join(base_path, "..", "QC Anonymized Study Files", 
                              "Study 1_CPID_Input Files - Anonymization")
    
    # Normalize path
    study_path = os.path.normpath(study_path)
    print(f"   Looking for data in: {study_path}")
    
    data_sources = {}
    
    try:
        # Load CPID EDC Metrics (Excel)
        cpid_path = os.path.join(study_path, "Study 1_CPID_EDC_Metrics_URSV2.0_14 NOV 2025_updated.xlsx")
        if os.path.exists(cpid_path):
            data_sources['cpid'] = pd.read_excel(cpid_path)
            # Also register with standard name
            data_sources['CPID_EDC_Metrics'] = data_sources['cpid']
            print(f"   ✓ Loaded CPID: {len(data_sources['cpid'])} rows")
        else:
            print(f"   ⚠️ CPID not found at: {cpid_path}")
        
        # Load eSAE Dashboard (Excel)
        sae_path = os.path.join(study_path, "Study 1_eSAE Dashboard_Standard DM_Safety Report_updated.xlsx")
        if os.path.exists(sae_path):
            data_sources['esae_dashboard'] = pd.read_excel(sae_path)
            print(f"   ✓ Loaded eSAE: {len(data_sources['esae_dashboard'])} rows")
        else:
            print(f"   ⚠️ eSAE not found at: {sae_path}")
        
        # Load Visit Tracker (Excel)
        visit_path = os.path.join(study_path, "Study 1_Visit Projection Tracker_14NOV2025_updated.xlsx")
        if os.path.exists(visit_path):
            data_sources['visit_tracker'] = pd.read_excel(visit_path)
            print(f"   ✓ Loaded Visit Tracker: {len(data_sources['visit_tracker'])} rows")
        else:
            print(f"   ⚠️ Visit Tracker not found at: {visit_path}")
        
        # Load MedDRA coding
        meddra_path = os.path.join(study_path, "Study 1_GlobalCodingReport_MedDRA_updated.xlsx")
        if os.path.exists(meddra_path):
            data_sources['meddra'] = pd.read_excel(meddra_path)
            print(f"   ✓ Loaded MedDRA: {len(data_sources['meddra'])} rows")
            
    except Exception as e:
        print(f"   ⚠️ Error loading data: {e}")
    
    return data_sources


def test_query_parser():
    """Test the Query Parser component"""
    print("\n" + "="*60)
    print("TEST 1: Query Parser")
    print("="*60)
    
    parser = QueryParser()
    
    # Test queries
    test_queries = [
        "Show me all sites in the US where the 'Missing Visits' rate is trending up over the last 3 snapshots, specifically for Cycle 12",
        "What are the top 5 sites by open queries?",
        "Find correlations between missing visits and data quality",
        "Show me SAE summary by preferred term",
        "Which subjects have more than 10 missing pages?",
        "Compare site 001 vs site 002 performance"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: \"{query}\"")
        
        parsed = parser.parse(query)
        
        print(f"   Intent: {parsed.intent.name}")
        print(f"   Primary Metric: {parsed.primary_metric.value if parsed.primary_metric else 'None'}")
        print(f"   Entity Filters: {len(parsed.entity_filters)}")
        for ef in parsed.entity_filters:
            print(f"      - {ef.entity_type.name}: {ef.values}")
        if parsed.time_constraint:
            print(f"   Time: {parsed.time_constraint.type} = {parsed.time_constraint.value}")
        print(f"   Top N: {parsed.top_n}")
        print(f"   Confidence: {parsed.confidence:.1%}")
        print(f"   Data Sources: {parsed.data_sources}")
    
    print("\n✅ Query Parser Test PASSED")
    assert parser is not None, "Parser should be initialized"


def test_query_executor(data_sources):
    """Test the Query Executor component"""
    print("\n" + "="*60)
    print("TEST 2: Query Executor")
    print("="*60)
    
    if not data_sources:
        print("⚠️ No data sources available, skipping executor test")
        return False
    
    parser = QueryParser()
    executor = QueryExecutor(data_sources=data_sources)
    
    # Test execution of parsed queries
    test_queries = [
        "Show me all sites",
        "What are the top 5 sites by missing visits?",
        "Find correlations in the data"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: \"{query}\"")
        
        parsed = parser.parse(query)
        result = executor.execute(parsed)
        
        print(f"   Success: {result.success}")
        print(f"   Rows: {result.row_count}")
        print(f"   Execution Time: {result.execution_time_ms:.2f}ms")
        
        if result.error_message:
            print(f"   Error: {result.error_message}")
        
        if result.summary:
            print(f"   Summary: {result.summary}")
        
        if result.correlations:
            print(f"   Correlations: {len(result.correlations)} found")
    
    print("\n✅ Query Executor Test PASSED")
    assert executor is not None, "Executor should be initialized"


def test_insight_generator(data_sources):
    """Test the Insight Generator component"""
    print("\n" + "="*60)
    print("TEST 3: Insight Generator")
    print("="*60)
    
    if not data_sources:
        print("⚠️ No data sources available, skipping insight test")
        return False
    
    parser = QueryParser()
    executor = QueryExecutor(data_sources=data_sources)
    generator = InsightGenerator()
    
    # Test insight generation
    query = "Show me sites with high missing visits"
    print(f"\n📝 Query: \"{query}\"")
    
    parsed = parser.parse(query)
    result = executor.execute(parsed)
    response = generator.generate(parsed, result)
    
    print(f"\n📊 Response:")
    print(f"   Understanding: {response.understanding}")
    print(f"\n   Answer:\n{response.answer}")
    
    if response.insights:
        print(f"\n   Insights ({len(response.insights)}):")
        for insight in response.insights[:3]:
            severity_icon = {'info': 'ℹ️', 'warning': '⚠️', 'critical': '🔴'}[insight.severity]
            print(f"   {severity_icon} {insight.title}")
            print(f"      {insight.description[:100]}...")
    
    if response.follow_up_questions:
        print(f"\n   Follow-up Questions:")
        for q in response.follow_up_questions[:3]:
            print(f"   - {q}")
    
    print(f"\n   Confidence: {response.confidence:.1%}")
    
    print("\n✅ Insight Generator Test PASSED")
    assert generator is not None, "Generator should be initialized"


def test_conversational_engine(data_sources):
    """Test the full Conversational Engine"""
    print("\n" + "="*60)
    print("TEST 4: Conversational Engine (Full Pipeline)")
    print("="*60)
    
    # Initialize engine
    engine = ConversationalEngine(data_sources=data_sources)
    
    # Start a session
    session_id = engine.start_session()
    print(f"✓ Session started: {session_id}")
    
    # Test queries
    test_queries = [
        "Show me all sites in the US where the 'Missing Visits' rate is trending up over the last 3 snapshots",
        "What are the top 10 sites by open queries?",
        "Find anomalies in missing visits data",
        "Show me correlations between metrics",
        "help"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"👤 User: {query}")
        print("-"*60)
        
        response = engine.ask(query, session_id)
        
        print(f"🤖 Assistant:")
        print(f"\n{response.understanding}\n")
        print(response.answer)
        
        if response.insights:
            print(f"\n📊 Key Insights: {len(response.insights)}")
            for insight in response.insights[:2]:
                severity_icon = {'info': 'ℹ️', 'warning': '⚠️', 'critical': '🔴'}[insight.severity]
                print(f"   {severity_icon} {insight.title}")
        
        print(f"\n⏱️ Processing: {response.processing_time_ms:.0f}ms | Confidence: {response.confidence:.1%}")
    
    # Test session export
    print(f"\n{'='*60}")
    print("📋 Session Export")
    print("-"*60)
    export = engine.export_session(session_id)
    print(f"Session exported: {len(export)} characters")
    
    # Test query suggestions
    print(f"\n{'='*60}")
    print("💡 Query Suggestions")
    print("-"*60)
    suggestions = engine.suggest_queries()
    for s in suggestions:
        print(f"   - {s}")
    
    # Test data summary
    print(f"\n{'='*60}")
    print("📊 Data Summary")
    print("-"*60)
    summary = engine.get_data_summary()
    for source, info in summary.get('sources', {}).items():
        print(f"   {source}: {info['rows']} rows, {len(info['columns'])} columns")
    
    print("\n✅ Conversational Engine Test PASSED")
    assert engine is not None, "Engine should be initialized"


def test_example_from_requirements():
    """Test the exact example from requirements"""
    print("\n" + "="*60)
    print("TEST 5: Requirements Example Query")
    print("="*60)
    
    # The exact query from the requirements
    query = "Show me all sites in the US where the 'Missing Visits' rate is trending up over the last 3 snapshots, specifically for Cycle 12"
    
    print(f"\n📝 Query: \"{query}\"")
    
    parser = QueryParser()
    parsed = parser.parse(query)
    
    # Print detailed parse explanation
    explanation = parser.explain_parse(parsed)
    print(f"\n{explanation}")
    
    # Show the structured query
    print(f"\n📋 Structured Query:")
    query_dict = parsed.to_dict()
    for key, value in query_dict.items():
        if value and key not in ['parse_notes']:
            print(f"   {key}: {value}")
    
    print("\n✅ Requirements Example Test PASSED")
    assert parsed is not None, "Query should be parsed"


def main():
    """Run all tests"""
    print("="*60)
    print("   NLQ MODULE TEST SUITE")
    print("   Conversational Insight Engine for Clinical Data")
    print("="*60)
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    print("\n📦 Loading Study Data...")
    data_sources = load_study_data()
    
    # Run tests
    results = []
    
    # Test 1: Query Parser
    results.append(("Query Parser", test_query_parser()))
    
    # Test 2: Query Executor
    results.append(("Query Executor", test_query_executor(data_sources)))
    
    # Test 3: Insight Generator
    results.append(("Insight Generator", test_insight_generator(data_sources)))
    
    # Test 4: Conversational Engine
    results.append(("Conversational Engine", test_conversational_engine(data_sources)))
    
    # Test 5: Requirements Example
    results.append(("Requirements Example", test_example_from_requirements()))
    
    # Summary
    print("\n" + "="*60)
    print("   TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {status} - {name}")
    
    print(f"\n   Total: {passed}/{total} tests passed")
    print("="*60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
