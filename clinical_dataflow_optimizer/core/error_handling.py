"""
Comprehensive Error Handling System
===================================

Provides centralized error handling, recovery mechanisms, and graceful degradation
for the Neural Clinical Data Mesh application.

Features:
- Custom exception hierarchy for clinical data errors
- Error boundaries for API endpoints
- Retry logic with exponential backoff
- Graceful degradation strategies
- Error logging and tracking
- Circuit breaker pattern implementation
"""

import logging
import traceback
import functools
import asyncio
from typing import Dict, Any, Optional, Callable, TypeVar, Generic, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import json
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# Custom Exception Hierarchy
# =============================================================================

class ClinicalDataError(Exception):
    """Base exception for all clinical data errors"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "CDM000",
        details: Dict = None,
        recoverable: bool = True
    ):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.recoverable = recoverable
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': str(self),
            'details': self.details,
            'recoverable': self.recoverable,
            'timestamp': self.timestamp.isoformat()
        }


class DataIngestionError(ClinicalDataError):
    """Errors during data ingestion"""
    
    def __init__(self, message: str, source: str = None, details: Dict = None):
        super().__init__(
            message,
            error_code="CDM100",
            details={'source': source, **(details or {})},
            recoverable=True
        )


class DataValidationError(ClinicalDataError):
    """Errors during data validation"""
    
    def __init__(self, message: str, field: str = None, value: Any = None, details: Dict = None):
        super().__init__(
            message,
            error_code="CDM200",
            details={'field': field, 'invalid_value': str(value), **(details or {})},
            recoverable=True
        )


class GraphProcessingError(ClinicalDataError):
    """Errors during knowledge graph processing"""
    
    def __init__(self, message: str, node_id: str = None, details: Dict = None):
        super().__init__(
            message,
            error_code="CDM300",
            details={'node_id': node_id, **(details or {})},
            recoverable=True
        )


class AgentExecutionError(ClinicalDataError):
    """Errors during agent execution"""
    
    def __init__(self, message: str, agent_name: str = None, details: Dict = None):
        super().__init__(
            message,
            error_code="CDM400",
            details={'agent_name': agent_name, **(details or {})},
            recoverable=True
        )


class LLMServiceError(ClinicalDataError):
    """Errors from LLM service"""
    
    def __init__(self, message: str, provider: str = None, details: Dict = None):
        super().__init__(
            message,
            error_code="CDM500",
            details={'provider': provider, **(details or {})},
            recoverable=True
        )


class APIError(ClinicalDataError):
    """Errors in API layer"""
    
    def __init__(self, message: str, endpoint: str = None, status_code: int = 500, details: Dict = None):
        super().__init__(
            message,
            error_code="CDM600",
            details={'endpoint': endpoint, 'status_code': status_code, **(details or {})},
            recoverable=True
        )


class ConfigurationError(ClinicalDataError):
    """Configuration errors"""
    
    def __init__(self, message: str, config_key: str = None, details: Dict = None):
        super().__init__(
            message,
            error_code="CDM700",
            details={'config_key': config_key, **(details or {})},
            recoverable=False
        )


class ResourceExhaustedError(ClinicalDataError):
    """Resource exhaustion errors"""
    
    def __init__(self, message: str, resource: str = None, details: Dict = None):
        super().__init__(
            message,
            error_code="CDM800",
            details={'resource': resource, **(details or {})},
            recoverable=True
        )


# =============================================================================
# Circuit Breaker Implementation
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = auto()    # Normal operation
    OPEN = auto()      # Blocking requests
    HALF_OPEN = auto() # Testing if service is back


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    recovery_timeout_seconds: int = 30
    half_open_max_calls: int = 3
    success_threshold: int = 2


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service is failing, requests are blocked
    - HALF_OPEN: Testing if service has recovered
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0
        self._lock = threading.Lock()
    
    def can_execute(self) -> bool:
        """Check if request can be executed"""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            elif self.state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self.last_failure_time:
                    elapsed = datetime.now() - self.last_failure_time
                    if elapsed.total_seconds() >= self.config.recovery_timeout_seconds:
                        self.state = CircuitState.HALF_OPEN
                        self.half_open_calls = 0
                        logger.info(f"Circuit {self.name} entering HALF_OPEN state")
                        return True
                return False
            else:  # HALF_OPEN
                return self.half_open_calls < self.config.half_open_max_calls
    
    def record_success(self):
        """Record successful execution"""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._close()
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0  # Reset failures on success
    
    def record_failure(self):
        """Record failed execution"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.state == CircuitState.HALF_OPEN:
                self._open()
            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self._open()
    
    def _open(self):
        """Open the circuit"""
        self.state = CircuitState.OPEN
        self.success_count = 0
        logger.warning(f"Circuit {self.name} OPENED after {self.failure_count} failures")
    
    def _close(self):
        """Close the circuit"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info(f"Circuit {self.name} CLOSED - service recovered")
    
    def get_state(self) -> Dict:
        """Get current circuit breaker state"""
        return {
            'name': self.name,
            'state': self.state.name,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure': self.last_failure_time.isoformat() if self.last_failure_time else None
        }


# Global circuit breaker registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
    """Get or create a circuit breaker"""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]


# =============================================================================
# Retry Logic with Exponential Backoff
# =============================================================================

@dataclass
class RetryConfig:
    """Configuration for retry logic"""
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (Exception,)


def retry_with_backoff(config: RetryConfig = None):
    """
    Decorator for retry with exponential backoff.
    
    Usage:
        @retry_with_backoff(RetryConfig(max_retries=3))
        def my_function():
            ...
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import random
            import time
            
            last_exception = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_retries:
                        logger.error(f"All {config.max_retries} retries failed for {func.__name__}")
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        config.base_delay_seconds * (config.exponential_base ** attempt),
                        config.max_delay_seconds
                    )
                    
                    # Add jitter
                    if config.jitter:
                        delay *= (0.5 + random.random())
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{config.max_retries} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator


async def async_retry_with_backoff(
    func: Callable,
    config: RetryConfig = None,
    *args,
    **kwargs
) -> Any:
    """Async version of retry with backoff"""
    import random
    
    if config is None:
        config = RetryConfig()
    
    last_exception = None
    
    for attempt in range(config.max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except config.retryable_exceptions as e:
            last_exception = e
            
            if attempt == config.max_retries:
                logger.error(f"All {config.max_retries} retries failed for {func.__name__}")
                raise
            
            delay = min(
                config.base_delay_seconds * (config.exponential_base ** attempt),
                config.max_delay_seconds
            )
            
            if config.jitter:
                delay *= (0.5 + random.random())
            
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
            await asyncio.sleep(delay)
    
    raise last_exception


# =============================================================================
# Graceful Degradation
# =============================================================================

@dataclass
class FallbackResult(Generic[T]):
    """Result with fallback indication"""
    value: T
    is_fallback: bool
    original_error: Optional[Exception] = None
    fallback_source: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'value': self.value if not callable(self.value) else str(self.value),
            'is_fallback': self.is_fallback,
            'original_error': str(self.original_error) if self.original_error else None,
            'fallback_source': self.fallback_source
        }


def with_fallback(fallback_value: Any = None, fallback_func: Callable = None):
    """
    Decorator for graceful degradation with fallback.
    
    Usage:
        @with_fallback(fallback_value={'status': 'unavailable'})
        def get_data():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return FallbackResult(value=result, is_fallback=False)
            except Exception as e:
                logger.warning(f"Function {func.__name__} failed, using fallback: {e}")
                
                if fallback_func:
                    try:
                        fallback_result = fallback_func(*args, **kwargs)
                        return FallbackResult(
                            value=fallback_result,
                            is_fallback=True,
                            original_error=e,
                            fallback_source='fallback_function'
                        )
                    except Exception as fallback_error:
                        logger.error(f"Fallback function also failed: {fallback_error}")
                
                return FallbackResult(
                    value=fallback_value,
                    is_fallback=True,
                    original_error=e,
                    fallback_source='default_value'
                )
        
        return wrapper
    return decorator


class GracefulDegradationManager:
    """
    Manages graceful degradation strategies for different services.
    """
    
    def __init__(self):
        self._strategies: Dict[str, Dict] = {}
        self._service_status: Dict[str, str] = {}
    
    def register_service(
        self,
        service_name: str,
        primary_func: Callable,
        fallback_func: Callable = None,
        fallback_value: Any = None,
        circuit_breaker: CircuitBreaker = None
    ):
        """Register a service with fallback strategy"""
        self._strategies[service_name] = {
            'primary': primary_func,
            'fallback_func': fallback_func,
            'fallback_value': fallback_value,
            'circuit_breaker': circuit_breaker or get_circuit_breaker(service_name)
        }
        self._service_status[service_name] = 'healthy'
    
    def execute(self, service_name: str, *args, **kwargs) -> FallbackResult:
        """Execute service with graceful degradation"""
        if service_name not in self._strategies:
            raise ValueError(f"Service {service_name} not registered")
        
        strategy = self._strategies[service_name]
        circuit_breaker = strategy['circuit_breaker']
        
        # Check circuit breaker
        if not circuit_breaker.can_execute():
            self._service_status[service_name] = 'degraded'
            return self._get_fallback_result(service_name, strategy, CircuitOpenError("Circuit is open"))
        
        # Try primary function
        try:
            result = strategy['primary'](*args, **kwargs)
            circuit_breaker.record_success()
            self._service_status[service_name] = 'healthy'
            return FallbackResult(value=result, is_fallback=False)
        except Exception as e:
            circuit_breaker.record_failure()
            self._service_status[service_name] = 'degraded'
            return self._get_fallback_result(service_name, strategy, e)
    
    def _get_fallback_result(
        self,
        service_name: str,
        strategy: Dict,
        original_error: Exception
    ) -> FallbackResult:
        """Get fallback result for failed service"""
        # Try fallback function
        if strategy['fallback_func']:
            try:
                fallback_result = strategy['fallback_func']()
                return FallbackResult(
                    value=fallback_result,
                    is_fallback=True,
                    original_error=original_error,
                    fallback_source=f'{service_name}_fallback_func'
                )
            except Exception as e:
                logger.error(f"Fallback function for {service_name} failed: {e}")
        
        # Return fallback value
        return FallbackResult(
            value=strategy['fallback_value'],
            is_fallback=True,
            original_error=original_error,
            fallback_source=f'{service_name}_fallback_value'
        )
    
    def get_status(self) -> Dict[str, str]:
        """Get status of all registered services"""
        return self._service_status.copy()


class CircuitOpenError(ClinicalDataError):
    """Error when circuit breaker is open"""
    
    def __init__(self, message: str = "Circuit breaker is open"):
        super().__init__(message, error_code="CDM900", recoverable=True)


# =============================================================================
# Error Logging and Tracking
# =============================================================================

@dataclass
class ErrorRecord:
    """Record of an error occurrence"""
    error_id: str
    error_type: str
    message: str
    details: Dict
    stack_trace: str
    timestamp: datetime
    context: Dict = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class ErrorTracker:
    """
    Tracks and manages error occurrences for monitoring and debugging.
    """
    
    def __init__(self, max_history: int = 1000):
        self._errors: List[ErrorRecord] = []
        self._error_counts: Dict[str, int] = {}
        self._lock = threading.Lock()
        self.max_history = max_history
    
    def record_error(
        self,
        error: Exception,
        context: Dict = None
    ) -> str:
        """Record an error occurrence"""
        import uuid
        
        error_id = f"err_{uuid.uuid4().hex[:12]}"
        
        record = ErrorRecord(
            error_id=error_id,
            error_type=type(error).__name__,
            message=str(error),
            details=error.details if isinstance(error, ClinicalDataError) else {},
            stack_trace=traceback.format_exc(),
            timestamp=datetime.now(),
            context=context or {}
        )
        
        with self._lock:
            self._errors.append(record)
            
            # Maintain history limit
            if len(self._errors) > self.max_history:
                self._errors = self._errors[-self.max_history:]
            
            # Update counts
            error_key = f"{record.error_type}:{record.message[:50]}"
            self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1
        
        logger.error(f"Error recorded [{error_id}]: {record.error_type} - {record.message}")
        return error_id
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict]:
        """Get recent error records"""
        with self._lock:
            recent = self._errors[-limit:]
            return [
                {
                    'error_id': e.error_id,
                    'error_type': e.error_type,
                    'message': e.message,
                    'timestamp': e.timestamp.isoformat(),
                    'resolved': e.resolved
                }
                for e in reversed(recent)
            ]
    
    def get_error_summary(self) -> Dict:
        """Get summary of error occurrences"""
        with self._lock:
            return {
                'total_errors': len(self._errors),
                'error_counts': dict(sorted(
                    self._error_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:20]),
                'unresolved': sum(1 for e in self._errors if not e.resolved)
            }
    
    def resolve_error(self, error_id: str):
        """Mark an error as resolved"""
        with self._lock:
            for error in self._errors:
                if error.error_id == error_id:
                    error.resolved = True
                    error.resolution_time = datetime.now()
                    break


# Global error tracker instance
_error_tracker: Optional[ErrorTracker] = None


def get_error_tracker() -> ErrorTracker:
    """Get or create global error tracker"""
    global _error_tracker
    if _error_tracker is None:
        _error_tracker = ErrorTracker()
    return _error_tracker


# =============================================================================
# API Error Handler
# =============================================================================

def api_error_handler(func: Callable) -> Callable:
    """
    Decorator for handling API endpoint errors.
    
    Catches exceptions and returns standardized error responses.
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ClinicalDataError as e:
            error_id = get_error_tracker().record_error(e)
            return {
                'success': False,
                'error': e.to_dict(),
                'error_id': error_id
            }
        except Exception as e:
            error_id = get_error_tracker().record_error(e)
            return {
                'success': False,
                'error': {
                    'error_type': type(e).__name__,
                    'message': str(e),
                    'error_code': 'CDM999'
                },
                'error_id': error_id
            }
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ClinicalDataError as e:
            error_id = get_error_tracker().record_error(e)
            return {
                'success': False,
                'error': e.to_dict(),
                'error_id': error_id
            }
        except Exception as e:
            error_id = get_error_tracker().record_error(e)
            return {
                'success': False,
                'error': {
                    'error_type': type(e).__name__,
                    'message': str(e),
                    'error_code': 'CDM999'
                },
                'error_id': error_id
            }
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


# =============================================================================
# Context Manager for Error Handling
# =============================================================================

@contextmanager
def error_boundary(
    operation_name: str,
    fallback_value: Any = None,
    reraise: bool = False
):
    """
    Context manager for error boundaries.
    
    Usage:
        with error_boundary('data_processing', fallback_value=[]):
            results = process_data(data)
    """
    try:
        yield
    except Exception as e:
        error_id = get_error_tracker().record_error(e, context={'operation': operation_name})
        logger.error(f"Error in {operation_name} [{error_id}]: {e}")
        
        if reraise:
            raise
        
        if fallback_value is not None:
            return fallback_value


# =============================================================================
# Health Check System
# =============================================================================

class ServiceHealth(Enum):
    """Service health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    service_name: str
    status: ServiceHealth
    latency_ms: float
    details: Dict = field(default_factory=dict)
    last_check: datetime = field(default_factory=datetime.now)


class HealthChecker:
    """
    Health check system for service monitoring.
    """
    
    def __init__(self):
        self._checks: Dict[str, Callable] = {}
        self._results: Dict[str, HealthCheckResult] = {}
    
    def register_check(self, service_name: str, check_func: Callable[[], bool]):
        """Register a health check function"""
        self._checks[service_name] = check_func
    
    def run_check(self, service_name: str) -> HealthCheckResult:
        """Run health check for a service"""
        import time
        
        if service_name not in self._checks:
            return HealthCheckResult(
                service_name=service_name,
                status=ServiceHealth.UNHEALTHY,
                latency_ms=0,
                details={'error': 'Service not registered'}
            )
        
        start_time = time.time()
        try:
            is_healthy = self._checks[service_name]()
            latency_ms = (time.time() - start_time) * 1000
            
            status = ServiceHealth.HEALTHY if is_healthy else ServiceHealth.UNHEALTHY
            
            # Check for degraded state (slow but working)
            if is_healthy and latency_ms > 1000:
                status = ServiceHealth.DEGRADED
            
            result = HealthCheckResult(
                service_name=service_name,
                status=status,
                latency_ms=latency_ms
            )
        except Exception as e:
            result = HealthCheckResult(
                service_name=service_name,
                status=ServiceHealth.UNHEALTHY,
                latency_ms=(time.time() - start_time) * 1000,
                details={'error': str(e)}
            )
        
        self._results[service_name] = result
        return result
    
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks"""
        for service_name in self._checks:
            self.run_check(service_name)
        return self._results.copy()
    
    def get_overall_status(self) -> ServiceHealth:
        """Get overall system health status"""
        if not self._results:
            return ServiceHealth.UNHEALTHY
        
        statuses = [r.status for r in self._results.values()]
        
        if all(s == ServiceHealth.HEALTHY for s in statuses):
            return ServiceHealth.HEALTHY
        elif any(s == ServiceHealth.UNHEALTHY for s in statuses):
            return ServiceHealth.UNHEALTHY
        else:
            return ServiceHealth.DEGRADED


# Global health checker
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get or create global health checker"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker
