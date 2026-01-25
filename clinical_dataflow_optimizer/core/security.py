"""
Security Hardening Module
=========================

Comprehensive security measures for the Clinical Dataflow Optimizer:
- Input validation and sanitization
- Authentication and authorization
- Rate limiting
- Audit logging
- Data encryption utilities
- HIPAA/GxP compliance helpers
"""

import logging
import hashlib
import hmac
import secrets
import re
import functools
import threading
from typing import Dict, Any, Optional, Callable, List, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from collections import defaultdict
import json
import base64

logger = logging.getLogger(__name__)


# =============================================================================
# Input Validation
# =============================================================================

class ValidationError(Exception):
    """Input validation error"""
    
    def __init__(self, field: str, message: str, value: Any = None):
        super().__init__(f"{field}: {message}")
        self.field = field
        self.message = message
        self.value = value


class InputValidator:
    """
    Input validation and sanitization for API requests.
    """
    
    # Patterns for common clinical data fields
    PATTERNS = {
        'study_id': r'^[A-Za-z0-9_-]{1,50}$',
        'patient_id': r'^[A-Za-z0-9_-]{1,100}$',
        'site_id': r'^[A-Za-z0-9_-]{1,50}$',
        'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'alphanumeric': r'^[A-Za-z0-9]+$',
        'safe_string': r'^[A-Za-z0-9\s\-_.,]+$',
    }
    
    # Dangerous characters to sanitize
    DANGEROUS_CHARS = ['<', '>', '"', "'", '&', '\x00', '\n', '\r']
    
    # SQL injection patterns
    SQL_PATTERNS = [
        r'(\s|^)(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE|EXEC|EXECUTE)(\s|$)',
        r'--',
        r';',
        r'/\*.*\*/',
    ]
    
    @classmethod
    def validate_pattern(cls, value: str, pattern_name: str, field_name: str) -> str:
        """Validate string against a named pattern"""
        if pattern_name not in cls.PATTERNS:
            raise ValueError(f"Unknown pattern: {pattern_name}")
        
        pattern = cls.PATTERNS[pattern_name]
        if not re.match(pattern, value):
            raise ValidationError(
                field_name,
                f"Value does not match required pattern for {pattern_name}",
                value
            )
        return value
    
    @classmethod
    def sanitize_string(cls, value: str) -> str:
        """Sanitize a string by removing dangerous characters"""
        if not isinstance(value, str):
            raise ValidationError("value", "Expected string", value)
        
        result = value
        for char in cls.DANGEROUS_CHARS:
            result = result.replace(char, '')
        
        return result.strip()
    
    @classmethod
    def validate_length(
        cls,
        value: str,
        field_name: str,
        min_length: int = 0,
        max_length: int = 10000
    ) -> str:
        """Validate string length"""
        if len(value) < min_length:
            raise ValidationError(
                field_name,
                f"Value too short (minimum {min_length} characters)",
                value
            )
        if len(value) > max_length:
            raise ValidationError(
                field_name,
                f"Value too long (maximum {max_length} characters)",
                value
            )
        return value
    
    @classmethod
    def check_sql_injection(cls, value: str, field_name: str) -> str:
        """Check for potential SQL injection patterns"""
        for pattern in cls.SQL_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"Potential SQL injection attempt in {field_name}")
                raise ValidationError(
                    field_name,
                    "Input contains potentially dangerous SQL patterns",
                    value
                )
        return value
    
    @classmethod
    def validate_clinical_id(cls, value: str, field_name: str) -> str:
        """Validate a clinical identifier (study/patient/site ID)"""
        value = cls.sanitize_string(value)
        cls.validate_length(value, field_name, 1, 100)
        cls.check_sql_injection(value, field_name)
        
        # Allow alphanumeric with dashes and underscores
        if not re.match(r'^[A-Za-z0-9_-]+$', value):
            raise ValidationError(
                field_name,
                "Clinical ID must contain only letters, numbers, dashes, and underscores",
                value
            )
        
        return value
    
    @classmethod
    def validate_query(cls, value: str, field_name: str = "query") -> str:
        """Validate a search/NLQ query"""
        value = cls.sanitize_string(value)
        cls.validate_length(value, field_name, 1, 1000)
        cls.check_sql_injection(value, field_name)
        return value


def validate_input(**validators):
    """
    Decorator for validating function inputs.
    
    Usage:
        @validate_input(
            study_id=('clinical_id', {'min_length': 1}),
            query=('query', {})
        )
        def search_study(study_id: str, query: str):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function argument names
            import inspect
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            # Combine args and kwargs
            bound_args = {}
            for i, arg in enumerate(args):
                if i < len(params):
                    bound_args[params[i]] = arg
            bound_args.update(kwargs)
            
            # Validate specified arguments
            for arg_name, (validator_type, options) in validators.items():
                if arg_name in bound_args:
                    value = bound_args[arg_name]
                    if value is not None:
                        if validator_type == 'clinical_id':
                            bound_args[arg_name] = InputValidator.validate_clinical_id(
                                value, arg_name
                            )
                        elif validator_type == 'query':
                            bound_args[arg_name] = InputValidator.validate_query(
                                value, arg_name
                            )
                        elif validator_type == 'pattern':
                            pattern = options.get('pattern', 'safe_string')
                            bound_args[arg_name] = InputValidator.validate_pattern(
                                value, pattern, arg_name
                            )
            
            return func(**bound_args)
        
        return wrapper
    return decorator


# =============================================================================
# Rate Limiting
# =============================================================================

@dataclass
class RateLimitRule:
    """Rate limit rule configuration"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_size: int = 10
    cooldown_seconds: int = 60


class RateLimiter:
    """
    Token bucket rate limiter with sliding window.
    """
    
    def __init__(self, rule: RateLimitRule = None):
        self.rule = rule or RateLimitRule()
        self._requests: Dict[str, List[datetime]] = defaultdict(list)
        self._blocked: Dict[str, datetime] = {}
        self._lock = threading.Lock()
    
    def _cleanup_old_requests(self, client_id: str, now: datetime):
        """Remove requests older than 1 hour"""
        cutoff = now - timedelta(hours=1)
        self._requests[client_id] = [
            t for t in self._requests[client_id] if t > cutoff
        ]
    
    def is_allowed(self, client_id: str) -> tuple[bool, Dict]:
        """Check if request is allowed under rate limits"""
        now = datetime.now()
        
        with self._lock:
            # Check if blocked
            if client_id in self._blocked:
                unblock_time = self._blocked[client_id]
                if now < unblock_time:
                    return False, {
                        'blocked': True,
                        'retry_after': (unblock_time - now).total_seconds()
                    }
                else:
                    del self._blocked[client_id]
            
            self._cleanup_old_requests(client_id, now)
            requests = self._requests[client_id]
            
            # Check requests in last minute
            minute_ago = now - timedelta(minutes=1)
            recent_minute = [t for t in requests if t > minute_ago]
            
            if len(recent_minute) >= self.rule.requests_per_minute:
                # Block for cooldown period
                self._blocked[client_id] = now + timedelta(seconds=self.rule.cooldown_seconds)
                logger.warning(f"Rate limit exceeded for {client_id}")
                return False, {
                    'exceeded': 'minute',
                    'retry_after': self.rule.cooldown_seconds
                }
            
            # Check requests in last hour
            hour_ago = now - timedelta(hours=1)
            recent_hour = [t for t in requests if t > hour_ago]
            
            if len(recent_hour) >= self.rule.requests_per_hour:
                return False, {
                    'exceeded': 'hour',
                    'retry_after': 3600 - (now - min(recent_hour)).total_seconds()
                }
            
            # Allow and record request
            self._requests[client_id].append(now)
            
            return True, {
                'remaining_minute': self.rule.requests_per_minute - len(recent_minute) - 1,
                'remaining_hour': self.rule.requests_per_hour - len(recent_hour) - 1
            }
    
    def get_status(self, client_id: str) -> Dict:
        """Get rate limit status for a client"""
        now = datetime.now()
        
        with self._lock:
            self._cleanup_old_requests(client_id, now)
            requests = self._requests[client_id]
            
            minute_ago = now - timedelta(minutes=1)
            hour_ago = now - timedelta(hours=1)
            
            recent_minute = len([t for t in requests if t > minute_ago])
            recent_hour = len([t for t in requests if t > hour_ago])
            
            return {
                'client_id': client_id,
                'requests_minute': recent_minute,
                'requests_hour': recent_hour,
                'limit_minute': self.rule.requests_per_minute,
                'limit_hour': self.rule.requests_per_hour,
                'blocked': client_id in self._blocked
            }


def rate_limit(limiter: RateLimiter = None, client_id_extractor: Callable = None):
    """
    Decorator for rate limiting function calls.
    
    Usage:
        @rate_limit(my_limiter, lambda args, kwargs: kwargs.get('user_id'))
        def api_endpoint(user_id: str):
            ...
    """
    _limiter = limiter or RateLimiter()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract client ID
            if client_id_extractor:
                client_id = client_id_extractor(args, kwargs)
            else:
                client_id = "default"
            
            allowed, info = _limiter.is_allowed(client_id)
            
            if not allowed:
                from core.error_handling import ResourceExhaustedError
                raise ResourceExhaustedError(
                    f"Rate limit exceeded. Retry after {info.get('retry_after', 60)} seconds",
                    resource="api_rate_limit",
                    details=info
                )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# =============================================================================
# Audit Logging
# =============================================================================

class AuditEventType(Enum):
    """Types of audit events"""
    LOGIN = "login"
    LOGOUT = "logout"
    ACCESS = "access"
    MODIFY = "modify"
    DELETE = "delete"
    EXPORT = "export"
    ADMIN = "admin"
    SECURITY = "security"
    QUERY = "query"


@dataclass
class AuditEvent:
    """An audit log event"""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: str
    action: str
    resource_type: str
    resource_id: Optional[str] = None
    details: Dict = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'action': self.action,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'details': self.details,
            'ip_address': self.ip_address,
            'success': self.success,
            'error_message': self.error_message
        }


class AuditLogger:
    """
    HIPAA/GxP compliant audit logging system.
    
    Maintains immutable audit trail with tamper detection.
    """
    
    def __init__(self, storage_path: str = None):
        self._events: List[AuditEvent] = []
        self._lock = threading.Lock()
        self._storage_path = storage_path
        self._hash_chain: List[str] = ["genesis"]  # Tamper detection chain
    
    def log(
        self,
        event_type: AuditEventType,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str = None,
        details: Dict = None,
        ip_address: str = None,
        user_agent: str = None,
        success: bool = True,
        error_message: str = None
    ) -> str:
        """Log an audit event"""
        import uuid
        
        event_id = f"audit_{uuid.uuid4().hex[:16]}"
        
        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message
        )
        
        with self._lock:
            # Add to chain for tamper detection
            previous_hash = self._hash_chain[-1]
            event_json = json.dumps(event.to_dict(), sort_keys=True)
            new_hash = hashlib.sha256(
                f"{previous_hash}{event_json}".encode()
            ).hexdigest()
            self._hash_chain.append(new_hash)
            
            self._events.append(event)
            
            # Persist to file if configured
            if self._storage_path:
                self._persist_event(event, new_hash)
        
        logger.info(f"Audit: [{event_type.value}] {user_id} - {action} on {resource_type}")
        return event_id
    
    def _persist_event(self, event: AuditEvent, hash_value: str):
        """Persist event to storage"""
        try:
            import os
            os.makedirs(self._storage_path, exist_ok=True)
            
            filename = event.timestamp.strftime("%Y-%m-%d") + ".jsonl"
            filepath = os.path.join(self._storage_path, filename)
            
            with open(filepath, 'a') as f:
                record = {**event.to_dict(), 'hash': hash_value}
                f.write(json.dumps(record) + '\n')
        except Exception as e:
            logger.error(f"Failed to persist audit event: {e}")
    
    def query(
        self,
        user_id: str = None,
        event_type: AuditEventType = None,
        resource_type: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 100
    ) -> List[Dict]:
        """Query audit log events"""
        with self._lock:
            results = []
            
            for event in reversed(self._events):
                if len(results) >= limit:
                    break
                
                if user_id and event.user_id != user_id:
                    continue
                if event_type and event.event_type != event_type:
                    continue
                if resource_type and event.resource_type != resource_type:
                    continue
                if start_time and event.timestamp < start_time:
                    continue
                if end_time and event.timestamp > end_time:
                    continue
                
                results.append(event.to_dict())
            
            return results
    
    def verify_integrity(self) -> bool:
        """Verify audit log integrity using hash chain"""
        with self._lock:
            if len(self._events) != len(self._hash_chain) - 1:
                return False
            
            current_hash = "genesis"
            for i, event in enumerate(self._events):
                event_json = json.dumps(event.to_dict(), sort_keys=True)
                computed_hash = hashlib.sha256(
                    f"{current_hash}{event_json}".encode()
                ).hexdigest()
                
                if computed_hash != self._hash_chain[i + 1]:
                    logger.error(f"Audit log integrity check failed at event {i}")
                    return False
                
                current_hash = computed_hash
            
            return True


def audit(
    event_type: AuditEventType,
    action: str,
    resource_type: str,
    audit_logger: AuditLogger = None
):
    """
    Decorator for automatic audit logging.
    
    Usage:
        @audit(AuditEventType.ACCESS, "view", "patient_data")
        def view_patient(user_id: str, patient_id: str):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _logger = audit_logger or get_audit_logger()
            user_id = kwargs.get('user_id', 'anonymous')
            resource_id = kwargs.get('resource_id') or kwargs.get('patient_id') or kwargs.get('study_id')
            
            try:
                result = func(*args, **kwargs)
                _logger.log(
                    event_type=event_type,
                    user_id=user_id,
                    action=action,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    success=True
                )
                return result
            except Exception as e:
                _logger.log(
                    event_type=event_type,
                    user_id=user_id,
                    action=action,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    success=False,
                    error_message=str(e)
                )
                raise
        
        return wrapper
    return decorator


# =============================================================================
# Authentication Helpers
# =============================================================================

class TokenManager:
    """
    Secure token generation and validation.
    """
    
    def __init__(self, secret_key: str = None, token_expiry_hours: int = 24):
        self._secret_key = secret_key or secrets.token_hex(32)
        self._token_expiry = timedelta(hours=token_expiry_hours)
        self._revoked_tokens: Set[str] = set()
    
    def generate_token(self, user_id: str, claims: Dict = None) -> str:
        """Generate a secure token"""
        import uuid
        
        payload = {
            'jti': uuid.uuid4().hex,
            'sub': user_id,
            'iat': datetime.now().isoformat(),
            'exp': (datetime.now() + self._token_expiry).isoformat(),
            **(claims or {})
        }
        
        payload_json = json.dumps(payload, sort_keys=True)
        payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).decode()
        
        # Create signature
        signature = hmac.new(
            self._secret_key.encode(),
            payload_b64.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return f"{payload_b64}.{signature}"
    
    def validate_token(self, token: str) -> Optional[Dict]:
        """Validate a token and return payload if valid"""
        try:
            parts = token.split('.')
            if len(parts) != 2:
                return None
            
            payload_b64, signature = parts
            
            # Verify signature
            expected_signature = hmac.new(
                self._secret_key.encode(),
                payload_b64.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                return None
            
            # Decode payload
            payload_json = base64.urlsafe_b64decode(payload_b64.encode()).decode()
            payload = json.loads(payload_json)
            
            # Check if revoked
            if payload.get('jti') in self._revoked_tokens:
                return None
            
            # Check expiry
            exp = datetime.fromisoformat(payload['exp'])
            if datetime.now() > exp:
                return None
            
            return payload
            
        except Exception as e:
            logger.warning(f"Token validation failed: {e}")
            return None
    
    def revoke_token(self, token: str):
        """Revoke a token"""
        payload = self.validate_token(token)
        if payload:
            self._revoked_tokens.add(payload.get('jti'))


class PermissionChecker:
    """
    Role-based access control (RBAC) permission checker.
    """
    
    # Default permission definitions
    PERMISSIONS = {
        'admin': {'read', 'write', 'delete', 'admin', 'export'},
        'data_manager': {'read', 'write', 'export'},
        'analyst': {'read', 'export'},
        'viewer': {'read'},
    }
    
    # Resource-specific permissions
    RESOURCE_PERMISSIONS = {
        'patient_data': {'read', 'write'},
        'study_config': {'read', 'write', 'admin'},
        'audit_logs': {'read'},
        'system_config': {'read', 'write', 'admin'},
    }
    
    @classmethod
    def has_permission(
        cls,
        role: str,
        permission: str,
        resource: str = None
    ) -> bool:
        """Check if a role has a specific permission"""
        role_perms = cls.PERMISSIONS.get(role, set())
        
        if permission not in role_perms:
            return False
        
        if resource and resource in cls.RESOURCE_PERMISSIONS:
            resource_perms = cls.RESOURCE_PERMISSIONS[resource]
            return permission in resource_perms
        
        return True
    
    @classmethod
    def require_permission(cls, permission: str, resource: str = None):
        """Decorator to require a specific permission"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                role = kwargs.get('user_role', 'viewer')
                
                if not cls.has_permission(role, permission, resource):
                    from core.error_handling import APIError
                    raise APIError(
                        f"Permission denied: {permission} on {resource or 'resource'}",
                        status_code=403
                    )
                
                return func(*args, **kwargs)
            
            return wrapper
        return decorator


# =============================================================================
# Data Encryption Utilities
# =============================================================================

class DataEncryption:
    """
    Utilities for encrypting sensitive data at rest.
    """
    
    def __init__(self, key: bytes = None):
        """Initialize with a 32-byte key (for AES-256)"""
        self._key = key or secrets.token_bytes(32)
    
    def encrypt(self, data: str) -> str:
        """Encrypt string data (simplified - use proper crypto library in production)"""
        # Note: In production, use cryptography library with AES-GCM
        # This is a simplified demonstration
        
        # XOR with key (NOT SECURE - for demo only)
        data_bytes = data.encode()
        key_extended = (self._key * (len(data_bytes) // len(self._key) + 1))[:len(data_bytes)]
        encrypted = bytes(a ^ b for a, b in zip(data_bytes, key_extended))
        
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt encrypted data"""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        key_extended = (self._key * (len(encrypted_bytes) // len(self._key) + 1))[:len(encrypted_bytes)]
        decrypted = bytes(a ^ b for a, b in zip(encrypted_bytes, key_extended))
        
        return decrypted.decode()
    
    @staticmethod
    def hash_pii(value: str, salt: str = "") -> str:
        """One-way hash for PII data (pseudonymization)"""
        return hashlib.sha256(f"{salt}{value}".encode()).hexdigest()


# =============================================================================
# Global Instances
# =============================================================================

_audit_logger: Optional[AuditLogger] = None
_rate_limiter: Optional[RateLimiter] = None
_token_manager: Optional[TokenManager] = None


def get_audit_logger() -> AuditLogger:
    """Get or create global audit logger"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def get_rate_limiter() -> RateLimiter:
    """Get or create global rate limiter"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def get_token_manager() -> TokenManager:
    """Get or create global token manager"""
    global _token_manager
    if _token_manager is None:
        _token_manager = TokenManager()
    return _token_manager
