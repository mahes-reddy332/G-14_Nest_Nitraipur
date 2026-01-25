"""
Circuit Breaker Implementation for API Resilience
"""

import asyncio
import time
import logging
from enum import Enum
from typing import Callable, Any, Optional
from dataclasses import dataclass
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class CircuitBreakerState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, requests rejected
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5  # Number of failures before opening
    recovery_timeout: float = 60.0  # Seconds to wait before trying again
    expected_exception: tuple = (Exception,)  # Exceptions that count as failures
    success_threshold: int = 3  # Successes needed in half-open state
    timeout: float = 10.0  # Request timeout

class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    pass

class CircuitBreaker:
    """Circuit breaker implementation for resilient API calls"""

    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self._lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        async with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' entering HALF_OPEN state")
                else:
                    raise CircuitBreakerOpenException(
                        f"Circuit breaker '{self.name}' is OPEN. Next retry in "
                        f"{self._time_until_reset():.1f} seconds"
                    )

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )

            async with self._lock:
                await self._on_success()

            return result

        except self.config.expected_exception as e:
            async with self._lock:
                await self._on_failure()
            raise e
        except asyncio.TimeoutError as e:
            async with self._lock:
                await self._on_failure()
            raise e

    async def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0

        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                logger.info(f"Circuit breaker '{self.name}' reset to CLOSED state")
        else:
            # Reset success count for closed state
            self.success_count = 0

    async def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker '{self.name}' failed in HALF_OPEN, returning to OPEN")
        elif self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker '{self.name}' opened after {self.failure_count} failures")

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.config.recovery_timeout

    def _time_until_reset(self) -> float:
        """Time remaining until next reset attempt"""
        if self.last_failure_time is None:
            return 0.0
        elapsed = time.time() - self.last_failure_time
        return max(0.0, self.config.recovery_timeout - elapsed)

    def get_status(self) -> dict:
        """Get current circuit breaker status"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "time_until_reset": self._time_until_reset(),
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout,
            }
        }

# Global circuit breaker registry
_circuit_breakers: dict[str, CircuitBreaker] = {}

def get_circuit_breaker(name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
    """Get or create a circuit breaker instance"""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]

def get_all_circuit_breakers() -> dict[str, dict]:
    """Get status of all circuit breakers"""
    return {name: cb.get_status() for name, cb in _circuit_breakers.items()}

@asynccontextmanager
async def circuit_breaker_context(name: str, config: CircuitBreakerConfig = None):
    """Context manager for circuit breaker usage"""
    cb = get_circuit_breaker(name, config)
    try:
        yield cb
    except Exception:
        raise
    finally:
        pass  # Circuit breaker state is managed internally</content>
<parameter name="filePath">d:\6932c39b908b6_detailed_problem_statements_and_datasets\Data for problem Statement 1\NEST 2.0 Data files_Anonymized\Main_Project Track1\clinical_dataflow_optimizer\api\utils\circuit_breaker.py