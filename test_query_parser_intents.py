#!/usr/bin/env python3
"""Test script for query parser intent recognition"""

from clinical_dataflow_optimizer.nlq.query_parser import QueryParser, QueryIntent

def test_new_intents():
    parser = QueryParser()

    test_queries = [
        'generate a patient safety narrative',
        'create an RBM report for site monitoring',
        'safety summary for patient 123',
        'generate CRA visit report',
        'patient safety narrative for subject 456',
        'create RBM monitoring report',
        'narrative for patient safety event'
    ]

    for query in test_queries:
        parsed = parser.parse(query)
        print(f'Query: "{query}"')
        print(f'Intent: {parsed.intent.name}')
        print(f'Confidence: {parsed.confidence:.3f}')
        print()

if __name__ == '__main__':
    test_new_intents()