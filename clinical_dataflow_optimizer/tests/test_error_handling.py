"""
Comprehensive Test Suite for Clinical Dataflow Optimizer
=========================================================

Tests for core functionality including:
- Error handling system
- LLM integration
- Inter-agent communication
- Digital twin processing
- NLQ/RAG processing
- Narrative generation
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

# Import core modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.error_handling import (
    ClinicalDataError,
    DataIngestionError,
    DataValidationError,
    GraphProcessingError,
    AgentExecutionError,
    LLMServiceError,
    APIError,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    RetryConfig,
    retry_with_backoff,
    with_fallback,
    FallbackResult,
    GracefulDegradationManager,
    ErrorTracker,
    get_error_tracker,
    HealthChecker,
    ServiceHealth,
    api_error_handler
)


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestCustomExceptions:
    """Tests for custom exception classes"""
    
    def test_clinical_data_error_creation(self):
        """Test base ClinicalDataError creation"""
        error = ClinicalDataError("Test error", error_code="TEST001")
        assert str(error) == "Test error"
        assert error.error_code == "TEST001"
        assert error.recoverable == True
        assert isinstance(error.timestamp, datetime)
    
    def test_clinical_data_error_to_dict(self):
        """Test error serialization"""
        error = ClinicalDataError("Test error", details={'key': 'value'})
        error_dict = error.to_dict()
        
        assert error_dict['error_type'] == 'ClinicalDataError'
        assert error_dict['message'] == 'Test error'
        assert error_dict['details']['key'] == 'value'
        assert 'timestamp' in error_dict
    
    def test_data_ingestion_error(self):
        """Test DataIngestionError with source"""
        error = DataIngestionError("Failed to ingest", source="CSV_FILE")
        assert error.error_code == "CDM100"
        assert error.details['source'] == "CSV_FILE"
    
    def test_data_validation_error(self):
        """Test DataValidationError with field and value"""
        error = DataValidationError("Invalid value", field="age", value=-5)
        assert error.error_code == "CDM200"
        assert error.details['field'] == "age"
        assert error.details['invalid_value'] == "-5"
    
    def test_graph_processing_error(self):
        """Test GraphProcessingError"""
        error = GraphProcessingError("Graph cycle detected", node_id="N001")
        assert error.error_code == "CDM300"
        assert error.details['node_id'] == "N001"
    
    def test_agent_execution_error(self):
        """Test AgentExecutionError"""
        error = AgentExecutionError("Agent timeout", agent_name="ProtocolAdvisor")
        assert error.error_code == "CDM400"
        assert error.details['agent_name'] == "ProtocolAdvisor"
    
    def test_llm_service_error(self):
        """Test LLMServiceError"""
        error = LLMServiceError("API rate limited", provider="openai")
        assert error.error_code == "CDM500"
        assert error.details['provider'] == "openai"


class TestCircuitBreaker:
    """Tests for circuit breaker pattern"""
    
    def test_circuit_starts_closed(self):
        """Test circuit breaker initial state"""
        cb = CircuitBreaker("test_service")
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute() == True
    
    def test_circuit_opens_after_failures(self):
        """Test circuit opens after threshold failures"""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test_service", config)
        
        # Record failures
        for _ in range(3):
            cb.record_failure()
        
        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() == False
    
    def test_circuit_half_open_after_timeout(self):
        """Test circuit enters half-open state after recovery timeout"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout_seconds=0  # Immediate recovery for testing
        )
        cb = CircuitBreaker("test_service", config)
        
        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        
        # Should transition to half-open
        assert cb.can_execute() == True
        assert cb.state == CircuitState.HALF_OPEN
    
    def test_circuit_closes_on_success_from_half_open(self):
        """Test circuit closes after successes in half-open state"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout_seconds=0,
            success_threshold=2
        )
        cb = CircuitBreaker("test_service", config)
        
        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        
        # Transition to half-open
        cb.can_execute()
        assert cb.state == CircuitState.HALF_OPEN
        
        # Record successes
        cb.record_success()
        cb.record_success()
        
        assert cb.state == CircuitState.CLOSED
    
    def test_circuit_get_state(self):
        """Test getting circuit breaker state"""
        cb = CircuitBreaker("test_service")
        state = cb.get_state()
        
        assert state['name'] == "test_service"
        assert state['state'] == "CLOSED"
        assert state['failure_count'] == 0


class TestRetryLogic:
    """Tests for retry with backoff"""
    
    def test_retry_success_first_attempt(self):
        """Test successful execution on first attempt"""
        call_count = 0
        
        @retry_with_backoff(RetryConfig(max_retries=3))
        def success_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = success_func()
        assert result == "success"
        assert call_count == 1
    
    def test_retry_success_after_failures(self):
        """Test success after initial failures"""
        call_count = 0
        
        @retry_with_backoff(RetryConfig(max_retries=3, base_delay_seconds=0.01))
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = flaky_func()
        assert result == "success"
        assert call_count == 3
    
    def test_retry_exhausted(self):
        """Test all retries exhausted"""
        config = RetryConfig(max_retries=2, base_delay_seconds=0.01)
        
        @retry_with_backoff(config)
        def always_fail():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            always_fail()


class TestFallback:
    """Tests for graceful degradation with fallback"""
    
    def test_fallback_not_used_on_success(self):
        """Test fallback not used when function succeeds"""
        @with_fallback(fallback_value="fallback")
        def success_func():
            return "primary"
        
        result = success_func()
        assert isinstance(result, FallbackResult)
        assert result.value == "primary"
        assert result.is_fallback == False
    
    def test_fallback_value_used_on_failure(self):
        """Test fallback value used when function fails"""
        @with_fallback(fallback_value="fallback")
        def fail_func():
            raise ValueError("Failure")
        
        result = fail_func()
        assert isinstance(result, FallbackResult)
        assert result.value == "fallback"
        assert result.is_fallback == True
        assert result.fallback_source == "default_value"
    
    def test_fallback_function_used(self):
        """Test fallback function is called on failure"""
        def backup_func(*args, **kwargs):
            return "backup_result"
        
        @with_fallback(fallback_func=backup_func)
        def fail_func():
            raise ValueError("Failure")
        
        result = fail_func()
        assert result.value == "backup_result"
        assert result.is_fallback == True
        assert result.fallback_source == "fallback_function"


class TestGracefulDegradationManager:
    """Tests for GracefulDegradationManager"""
    
    def test_register_and_execute_service(self):
        """Test registering and executing a service"""
        manager = GracefulDegradationManager()
        
        manager.register_service(
            "test_service",
            primary_func=lambda: "primary_result",
            fallback_value="fallback_result"
        )
        
        result = manager.execute("test_service")
        assert result.value == "primary_result"
        assert result.is_fallback == False
    
    def test_fallback_on_primary_failure(self):
        """Test fallback is used when primary fails"""
        manager = GracefulDegradationManager()
        
        def failing_primary():
            raise ValueError("Primary failed")
        
        manager.register_service(
            "test_service",
            primary_func=failing_primary,
            fallback_value="fallback_result"
        )
        
        # Need to fail enough times to trigger fallback through circuit breaker
        for _ in range(6):  # Exceed failure threshold
            result = manager.execute("test_service")
        
        assert result.is_fallback == True
    
    def test_get_status(self):
        """Test getting service status"""
        manager = GracefulDegradationManager()
        
        manager.register_service(
            "service_a",
            primary_func=lambda: "result",
            fallback_value=None
        )
        
        manager.execute("service_a")
        status = manager.get_status()
        
        assert "service_a" in status


class TestErrorTracker:
    """Tests for error tracking"""
    
    def test_record_error(self):
        """Test recording an error"""
        tracker = ErrorTracker()
        error = ClinicalDataError("Test error")
        
        error_id = tracker.record_error(error)
        
        assert error_id.startswith("err_")
        recent = tracker.get_recent_errors(1)
        assert len(recent) == 1
        assert recent[0]['message'] == "Test error"
    
    def test_error_summary(self):
        """Test error summary generation"""
        tracker = ErrorTracker()
        
        # Record multiple errors
        for i in range(5):
            tracker.record_error(ClinicalDataError(f"Error {i % 2}"))
        
        summary = tracker.get_error_summary()
        assert summary['total_errors'] == 5
        assert summary['unresolved'] == 5
    
    def test_resolve_error(self):
        """Test resolving an error"""
        tracker = ErrorTracker()
        error = ClinicalDataError("Test error")
        
        error_id = tracker.record_error(error)
        tracker.resolve_error(error_id)
        
        summary = tracker.get_error_summary()
        assert summary['unresolved'] == 0
    
    def test_max_history_limit(self):
        """Test that history is limited"""
        tracker = ErrorTracker(max_history=10)
        
        # Record more than max
        for i in range(20):
            tracker.record_error(ClinicalDataError(f"Error {i}"))
        
        summary = tracker.get_error_summary()
        assert summary['total_errors'] == 10


class TestHealthChecker:
    """Tests for health check system"""
    
    def test_register_and_run_check(self):
        """Test registering and running health check"""
        checker = HealthChecker()
        
        checker.register_check("test_service", lambda: True)
        result = checker.run_check("test_service")
        
        assert result.service_name == "test_service"
        assert result.status == ServiceHealth.HEALTHY
    
    def test_unhealthy_check(self):
        """Test unhealthy check result"""
        checker = HealthChecker()
        
        checker.register_check("test_service", lambda: False)
        result = checker.run_check("test_service")
        
        assert result.status == ServiceHealth.UNHEALTHY
    
    def test_check_with_exception(self):
        """Test check that raises exception"""
        checker = HealthChecker()
        
        def failing_check():
            raise RuntimeError("Service down")
        
        checker.register_check("test_service", failing_check)
        result = checker.run_check("test_service")
        
        assert result.status == ServiceHealth.UNHEALTHY
        assert 'error' in result.details
    
    def test_overall_status(self):
        """Test overall system status"""
        checker = HealthChecker()
        
        checker.register_check("service_a", lambda: True)
        checker.register_check("service_b", lambda: True)
        
        checker.run_all_checks()
        assert checker.get_overall_status() == ServiceHealth.HEALTHY
        
        # Add an unhealthy service
        checker.register_check("service_c", lambda: False)
        checker.run_all_checks()
        assert checker.get_overall_status() == ServiceHealth.UNHEALTHY


class TestAPIErrorHandler:
    """Tests for API error handler decorator"""
    
    def test_successful_execution(self):
        """Test decorator with successful execution"""
        @api_error_handler
        def success_func():
            return {"data": "value"}
        
        result = success_func()
        assert result == {"data": "value"}
    
    def test_clinical_error_handling(self):
        """Test decorator with ClinicalDataError"""
        @api_error_handler
        def fail_func():
            raise DataValidationError("Invalid data", field="test")
        
        result = fail_func()
        assert result['success'] == False
        assert result['error']['error_code'] == "CDM200"
        assert 'error_id' in result
    
    def test_generic_error_handling(self):
        """Test decorator with generic exception"""
        @api_error_handler
        def fail_func():
            raise RuntimeError("Unexpected error")
        
        result = fail_func()
        assert result['success'] == False
        assert result['error']['error_type'] == "RuntimeError"


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for error handling with other components"""
    
    def test_circuit_breaker_with_retry(self):
        """Test circuit breaker integrated with retry logic"""
        from core.error_handling import get_circuit_breaker
        
        cb = get_circuit_breaker("integration_test", CircuitBreakerConfig(failure_threshold=3))
        call_count = 0
        
        @retry_with_backoff(RetryConfig(max_retries=2, base_delay_seconds=0.01))
        def protected_call():
            nonlocal call_count
            call_count += 1
            if not cb.can_execute():
                raise RuntimeError("Circuit open")
            try:
                raise ValueError("Service error")
            except ValueError:
                cb.record_failure()
                raise
        
        with pytest.raises(ValueError):
            protected_call()
        
        # Should have retried
        assert call_count >= 2
    
    def test_error_tracking_with_fallback(self):
        """Test error tracking integrated with fallback"""
        tracker = ErrorTracker()
        
        @with_fallback(fallback_value="safe_default")
        def tracked_operation():
            error = LLMServiceError("Rate limited", provider="test")
            tracker.record_error(error)
            raise error
        
        result = tracked_operation()
        
        assert result.is_fallback == True
        assert tracker.get_error_summary()['total_errors'] == 1


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance tests for error handling components"""
    
    def test_error_tracker_performance(self):
        """Test error tracker can handle high volume"""
        import time
        
        tracker = ErrorTracker(max_history=1000)
        
        start = time.time()
        for i in range(1000):
            tracker.record_error(ClinicalDataError(f"Error {i}"))
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 1.0  # Less than 1 second for 1000 errors
        assert tracker.get_error_summary()['total_errors'] == 1000
    
    def test_circuit_breaker_thread_safety(self):
        """Test circuit breaker is thread-safe"""
        import threading
        
        cb = CircuitBreaker("thread_test", CircuitBreakerConfig(failure_threshold=100))
        
        def record_failures():
            for _ in range(50):
                cb.record_failure()
        
        threads = [threading.Thread(target=record_failures) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All failures should be recorded
        assert cb.failure_count == 200


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def error_tracker():
    """Provide fresh error tracker for each test"""
    return ErrorTracker()


@pytest.fixture
def circuit_breaker():
    """Provide fresh circuit breaker for each test"""
    return CircuitBreaker("test_circuit")


@pytest.fixture
def health_checker():
    """Provide fresh health checker for each test"""
    return HealthChecker()


@pytest.fixture
def degradation_manager():
    """Provide fresh degradation manager for each test"""
    return GracefulDegradationManager()


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
