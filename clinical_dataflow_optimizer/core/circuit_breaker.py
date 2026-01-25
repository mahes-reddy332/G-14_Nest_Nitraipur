"""
Circuit Breaker Pattern for Neural Clinical Data Mesh
Implements resilient API calls with failure detection and recovery
"""

import time
import threading
from enum import Enum
from typing import Callable, Any, Optional, Dict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    """
    Circuit breaker implementation for resilient API calls
    """

    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: Exception = Exception,
                 name: str = "CircuitBreaker"):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.lock = threading.Lock()

        logger.info(f"🚀 Circuit breaker '{name}' initialized")

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit"""
        if self.last_failure_time is None:
            return False

        elapsed = time.time() - self.last_failure_time
        return elapsed >= self.recovery_timeout

    def _record_success(self):
        """Record a successful call"""
        with self.lock:
            self.failure_count = 0
            self.state = CircuitState.CLOSED
            logger.info(f"✅ Circuit breaker '{self.name}' success - CLOSED")

    def _record_failure(self):
        """Record a failed call"""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"🔴 Circuit breaker '{self.name}' OPEN after {self.failure_count} failures")
            else:
                logger.warning(f"⚠️  Circuit breaker '{self.name}' failure {self.failure_count}/{self.failure_threshold}")

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result or fallback value

        Raises:
            CircuitBreakerOpenException: If circuit is open
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                with self.lock:
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"🔄 Circuit breaker '{self.name}' testing recovery - HALF_OPEN")
            else:
                raise CircuitBreakerOpenException(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Next retry in {self.recovery_timeout - (time.time() - self.last_failure_time):.1f}s"
                )

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result

        except self.expected_exception as e:
            self._record_failure()

            if self.state == CircuitState.HALF_OPEN:
                # Test failed, go back to open
                with self.lock:
                    self.state = CircuitState.OPEN
                    logger.warning(f"❌ Circuit breaker '{self.name}' recovery test failed - OPEN")

            raise e

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'failure_threshold': self.failure_threshold,
            'last_failure_time': self.last_failure_time,
            'recovery_timeout': self.recovery_timeout,
            'time_until_retry': max(0, self.recovery_timeout - (time.time() - (self.last_failure_time or 0)))
        }

class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    pass

# Global circuit breakers for different services
circuit_breakers = {
    'longcat_api': CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=30,
        name='LongCat API'
    ),
    'graph_queries': CircuitBreaker(
        failure_threshold=5,
        recovery_timeout=60,
        name='Graph Queries'
    ),
    'feature_engineering': CircuitBreaker(
        failure_threshold=2,
        recovery_timeout=15,
        name='Feature Engineering'
    ),
    'external_apis': CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=45,
        name='External APIs'
    )
}

def with_circuit_breaker(breaker_name: str):
    """
    Decorator to apply circuit breaker to a function

    Args:
        breaker_name: Name of circuit breaker to use
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            breaker = circuit_breakers.get(breaker_name)
            if breaker:
                return breaker.call(func, *args, **kwargs)
            else:
                # No circuit breaker configured, execute normally
                return func(*args, **kwargs)
        return wrapper
    return decorator

def get_circuit_breaker_status() -> Dict[str, Dict[str, Any]]:
    """Get status of all circuit breakers"""
    return {name: breaker.get_status() for name, breaker in circuit_breakers.items()}

def reset_circuit_breaker(name: str) -> bool:
    """Manually reset a circuit breaker"""
    if name in circuit_breakers:
        breaker = circuit_breakers[name]
        with breaker.lock:
            breaker.state = CircuitState.CLOSED
            breaker.failure_count = 0
            breaker.last_failure_time = None
        logger.info(f"🔧 Circuit breaker '{name}' manually reset")
        return True
    return False