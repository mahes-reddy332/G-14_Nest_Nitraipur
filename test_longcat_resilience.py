#!/usr/bin/env python3
"""
Test script for enhanced LongCat integration with resilience patterns
Tests circuit breaker, caching, graceful degradation, and timeout handling
"""

import sys
import os
import time
import json
from pathlib import Path

# Import from the clinical_dataflow_optimizer package
from clinical_dataflow_optimizer.core.longcat_integration import LongCatClient

def test_circuit_breaker():
    """Test circuit breaker functionality"""
    print("Testing Circuit Breaker...")

    client = LongCatClient()

    # Test initial state
    assert not client.circuit_breaker.is_open(), "Circuit breaker should start closed"
    print("✓ Circuit breaker starts closed")

    # Simulate failures
    for i in range(client.circuit_breaker.failure_threshold):
        client.circuit_breaker.record_failure()

    assert client.circuit_breaker.is_open(), "Circuit breaker should open after failures"
    print("✓ Circuit breaker opens after failure threshold")

    # Test recovery - circuit breaker should allow attempt after timeout
    time.sleep(client.circuit_breaker.recovery_timeout + 1)
    assert client.circuit_breaker.should_attempt_request(), "Circuit breaker should allow attempt after recovery timeout"
    print("✓ Circuit breaker allows attempts after recovery timeout")

    # Simulate successful request to fully recover
    client.circuit_breaker.record_success()
    assert not client.circuit_breaker.is_open(), "Circuit breaker should close after success"
    print("✓ Circuit breaker recovers fully after successful request")

def test_caching():
    """Test response caching functionality"""
    print("\nTesting Response Caching...")

    client = LongCatClient()

    # Test cache methods exist
    assert hasattr(client, '_get_cached_response'), "Cache get method should exist"
    assert hasattr(client, '_cache_response'), "Cache set method should exist"
    print("✓ Cache methods exist")

    # Test cache operations
    test_key = "test_key"
    test_response = {"test": "data"}

    # Cache a response
    client._cache_response(test_key, test_response)

    # Retrieve cached response
    cached = client._get_cached_response(test_key)
    assert cached == test_response, "Cached response should match"
    print("✓ Cache store and retrieve works")

def test_graceful_degradation():
    """Test graceful degradation when LongCat is unavailable"""
    print("\nTesting Graceful Degradation...")

    client = LongCatClient()

    # Test fallback narrative generation
    test_patient_data = {
        'subject_id': 'TEST001',
        'status': 'Active',
        'clean_status': True
    }
    test_issues = ['Missing data', 'Outlier detected']
    test_recommendations = ['Review data', 'Contact site']

    fallback_narrative = client._generate_fallback_narrative(
        test_patient_data, test_issues, test_recommendations
    )

    assert 'TEST001' in fallback_narrative, "Fallback narrative should contain patient ID"
    assert 'Automated Analysis' in fallback_narrative, "Should indicate automated analysis"
    print("✓ Fallback narrative generation works")

    # Test fallback anomaly explanation
    test_anomaly = {
        'type': 'Data Inconsistency',
        'severity': 'High',
        'description': 'Conflicting values detected'
    }
    test_context = "Patient data validation"

    fallback_explanation = client._generate_fallback_anomaly_explanation(
        test_anomaly, test_context
    )

    assert 'Data Inconsistency' in fallback_explanation, "Should contain anomaly type"
    assert 'Automated Analysis' in fallback_explanation, "Should indicate automated analysis"
    print("✓ Fallback anomaly explanation works")

def test_payload_truncation():
    """Test payload truncation for large inputs"""
    print("\nTesting Payload Truncation...")

    client = LongCatClient()

    # Test large patient data truncation in narrative generation
    large_patient_data = {
        'subject_id': 'TEST001',
        'status': 'Active',
        'clean_status': True,
        'large_field': 'x' * 5000  # Large field that should be truncated
    }

    # This should work without errors (graceful degradation if LongCat unavailable)
    try:
        result = client.generate_narrative(large_patient_data, [], [])
        assert isinstance(result, str), "Should return string result"
        print("✓ Large payload handling works")
    except Exception as e:
        # If LongCat is unavailable, it should use fallback
        if "Circuit breaker is open" in str(e) or "LongCat service unavailable" in str(e):
            print("✓ Circuit breaker properly blocks requests when open")
        else:
            raise e

def test_timeout_configuration():
    """Test timeout configuration"""
    print("\nTesting Timeout Configuration...")

    client = LongCatClient()

    # Check that timeout config exists
    assert hasattr(client.config, 'connect_timeout'), "Should have connect timeout"
    assert hasattr(client.config, 'read_timeout'), "Should have read timeout"
    assert client.config.connect_timeout == 10, "Connect timeout should be 10s"
    assert client.config.read_timeout == 45, "Read timeout should be 45s"
    print("✓ Timeout configuration is correct")

def run_all_tests():
    """Run all resilience tests"""
    print("Running LongCat Integration Resilience Tests")
    print("=" * 50)

    try:
        test_circuit_breaker()
        test_caching()
        test_graceful_degradation()
        test_payload_truncation()
        test_timeout_configuration()

        print("\n" + "=" * 50)
        print("✅ All resilience tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)