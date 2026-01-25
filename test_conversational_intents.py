#!/usr/bin/env python3
"""Test script for conversational engine with new intents"""

from clinical_dataflow_optimizer.nlq.conversational_engine import ConversationalEngine

def test_new_intents():
    # Initialize engine (without data for now)
    engine = ConversationalEngine()

    # Test narrative generation
    print("Testing NARRATIVE_GENERATION intent:")
    response1 = engine.ask("generate a patient safety narrative")
    print(f"Response: {response1.answer[:200]}...")
    print(f"Confidence: {response1.confidence}")
    print()

    # Test RBM report generation
    print("Testing RBM_REPORT intent:")
    response2 = engine.ask("create an RBM report for site monitoring")
    print(f"Response: {response2.answer[:200]}...")
    print(f"Confidence: {response2.confidence}")
    print()

if __name__ == '__main__':
    test_new_intents()