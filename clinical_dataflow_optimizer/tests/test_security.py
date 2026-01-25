"""
Test Suite for Security Module
==============================

Tests for:
- Input validation
- Rate limiting
- Audit logging
- Authentication helpers
"""

import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.security import (
    ValidationError,
    InputValidator,
    validate_input,
    RateLimiter,
    RateLimitRule,
    rate_limit,
    AuditLogger,
    AuditEventType,
    audit,
    TokenManager,
    PermissionChecker,
    DataEncryption,
    get_audit_logger,
    get_rate_limiter,
    get_token_manager
)


# =============================================================================
# Input Validation Tests
# =============================================================================

class TestInputValidator:
    """Tests for InputValidator"""
    
    def test_validate_pattern_alphanumeric(self):
        """Test alphanumeric pattern validation"""
        result = InputValidator.validate_pattern("ABC123", "alphanumeric", "test_field")
        assert result == "ABC123"
    
    def test_validate_pattern_invalid(self):
        """Test pattern validation with invalid input"""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_pattern("ABC 123!", "alphanumeric", "test_field")
        assert exc_info.value.field == "test_field"
    
    def test_validate_pattern_unknown(self):
        """Test unknown pattern raises error"""
        with pytest.raises(ValueError):
            InputValidator.validate_pattern("test", "unknown_pattern", "test_field")
    
    def test_sanitize_string(self):
        """Test string sanitization"""
        result = InputValidator.sanitize_string("Hello<script>alert('xss')</script>World")
        assert '<' not in result
        assert '>' not in result
    
    def test_sanitize_string_non_string(self):
        """Test sanitization with non-string raises error"""
        with pytest.raises(ValidationError):
            InputValidator.sanitize_string(123)
    
    def test_validate_length_valid(self):
        """Test length validation with valid input"""
        result = InputValidator.validate_length("hello", "test", min_length=1, max_length=10)
        assert result == "hello"
    
    def test_validate_length_too_short(self):
        """Test length validation with short input"""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_length("hi", "test", min_length=5)
        assert "too short" in str(exc_info.value)
    
    def test_validate_length_too_long(self):
        """Test length validation with long input"""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_length("a" * 100, "test", max_length=10)
        assert "too long" in str(exc_info.value)
    
    def test_check_sql_injection(self):
        """Test SQL injection detection"""
        # Should pass normal input
        result = InputValidator.check_sql_injection("normal search query", "query")
        assert result == "normal search query"
        
        # Should catch SQL injection
        with pytest.raises(ValidationError):
            InputValidator.check_sql_injection("'; DROP TABLE patients; --", "query")
    
    def test_check_sql_injection_select(self):
        """Test SQL injection with SELECT"""
        with pytest.raises(ValidationError):
            InputValidator.check_sql_injection("UNION SELECT * FROM users", "query")
    
    def test_validate_clinical_id_valid(self):
        """Test clinical ID validation with valid input"""
        result = InputValidator.validate_clinical_id("STUDY-001", "study_id")
        assert result == "STUDY-001"
    
    def test_validate_clinical_id_invalid(self):
        """Test clinical ID validation with invalid characters"""
        with pytest.raises(ValidationError):
            InputValidator.validate_clinical_id("STUDY@001!", "study_id")
    
    def test_validate_query(self):
        """Test query validation"""
        result = InputValidator.validate_query("What is the status of patient 123?")
        assert len(result) > 0


class TestValidateInputDecorator:
    """Tests for validate_input decorator"""
    
    def test_decorator_validates_arguments(self):
        """Test decorator validates specified arguments"""
        @validate_input(study_id=('clinical_id', {}))
        def test_func(study_id: str):
            return study_id
        
        result = test_func(study_id="STUDY-001")
        assert result == "STUDY-001"
    
    def test_decorator_rejects_invalid(self):
        """Test decorator rejects invalid input"""
        @validate_input(study_id=('clinical_id', {}))
        def test_func(study_id: str):
            return study_id
        
        with pytest.raises(ValidationError):
            test_func(study_id="STUDY@001!")


# =============================================================================
# Rate Limiting Tests
# =============================================================================

class TestRateLimiter:
    """Tests for RateLimiter"""
    
    def test_allows_under_limit(self):
        """Test allows requests under limit"""
        rule = RateLimitRule(requests_per_minute=10, requests_per_hour=100)
        limiter = RateLimiter(rule)
        
        for i in range(5):
            allowed, info = limiter.is_allowed("client1")
            assert allowed == True
    
    def test_blocks_over_minute_limit(self):
        """Test blocks requests over minute limit"""
        rule = RateLimitRule(requests_per_minute=5, requests_per_hour=100, cooldown_seconds=1)
        limiter = RateLimiter(rule)
        
        # Make requests up to limit
        for i in range(5):
            allowed, _ = limiter.is_allowed("client1")
        
        # Next request should be blocked
        allowed, info = limiter.is_allowed("client1")
        assert allowed == False
        assert 'retry_after' in info
    
    def test_separate_clients(self):
        """Test separate limits per client"""
        rule = RateLimitRule(requests_per_minute=3, requests_per_hour=100)
        limiter = RateLimiter(rule)
        
        # Exhaust limit for client1
        for i in range(3):
            limiter.is_allowed("client1")
        
        # client2 should still be allowed
        allowed, _ = limiter.is_allowed("client2")
        assert allowed == True
    
    def test_get_status(self):
        """Test getting rate limit status"""
        rule = RateLimitRule(requests_per_minute=10, requests_per_hour=100)
        limiter = RateLimiter(rule)
        
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")
        
        status = limiter.get_status("client1")
        
        assert status['client_id'] == 'client1'
        assert status['requests_minute'] == 2
        assert status['limit_minute'] == 10


class TestRateLimitDecorator:
    """Tests for rate_limit decorator"""
    
    def test_decorator_allows_requests(self):
        """Test decorator allows requests under limit"""
        limiter = RateLimiter(RateLimitRule(requests_per_minute=10))
        
        @rate_limit(limiter=limiter, client_id_extractor=lambda a, k: k.get('user_id', 'default'))
        def test_func(user_id: str):
            return f"Hello, {user_id}"
        
        result = test_func(user_id="user1")
        assert result == "Hello, user1"


# =============================================================================
# Audit Logger Tests
# =============================================================================

class TestAuditLogger:
    """Tests for AuditLogger"""
    
    def test_log_event(self):
        """Test logging an audit event"""
        logger = AuditLogger()
        
        event_id = logger.log(
            event_type=AuditEventType.ACCESS,
            user_id="user123",
            action="view",
            resource_type="patient_data",
            resource_id="patient456"
        )
        
        assert event_id.startswith("audit_")
    
    def test_query_events(self):
        """Test querying audit events"""
        logger = AuditLogger()
        
        logger.log(
            event_type=AuditEventType.ACCESS,
            user_id="user123",
            action="view",
            resource_type="patient_data"
        )
        logger.log(
            event_type=AuditEventType.MODIFY,
            user_id="user456",
            action="update",
            resource_type="study_config"
        )
        
        results = logger.query(user_id="user123")
        assert len(results) == 1
        assert results[0]['user_id'] == "user123"
    
    def test_query_by_event_type(self):
        """Test querying by event type"""
        logger = AuditLogger()
        
        logger.log(
            event_type=AuditEventType.ACCESS,
            user_id="user1",
            action="view",
            resource_type="data"
        )
        logger.log(
            event_type=AuditEventType.MODIFY,
            user_id="user2",
            action="edit",
            resource_type="data"
        )
        
        results = logger.query(event_type=AuditEventType.MODIFY)
        assert len(results) == 1
        assert results[0]['event_type'] == 'modify'
    
    def test_verify_integrity(self):
        """Test audit log integrity verification"""
        logger = AuditLogger()
        
        logger.log(
            event_type=AuditEventType.ACCESS,
            user_id="user1",
            action="view",
            resource_type="data"
        )
        logger.log(
            event_type=AuditEventType.MODIFY,
            user_id="user2",
            action="edit",
            resource_type="data"
        )
        
        assert logger.verify_integrity() == True
    
    def test_integrity_tampering_detected(self):
        """Test that tampering is detected"""
        logger = AuditLogger()
        
        logger.log(
            event_type=AuditEventType.ACCESS,
            user_id="user1",
            action="view",
            resource_type="data"
        )
        
        # Tamper with hash chain
        logger._hash_chain[-1] = "tampered_hash"
        
        assert logger.verify_integrity() == False


class TestAuditDecorator:
    """Tests for audit decorator"""
    
    def test_decorator_logs_success(self):
        """Test decorator logs successful operations"""
        logger = AuditLogger()
        
        @audit(AuditEventType.ACCESS, "view", "patient_data", audit_logger=logger)
        def view_patient(user_id: str, patient_id: str):
            return {"patient_id": patient_id}
        
        view_patient(user_id="user123", patient_id="patient456")
        
        events = logger.query()
        assert len(events) == 1
        assert events[0]['success'] == True
    
    def test_decorator_logs_failure(self):
        """Test decorator logs failed operations"""
        logger = AuditLogger()
        
        @audit(AuditEventType.ACCESS, "view", "patient_data", audit_logger=logger)
        def view_patient(user_id: str, patient_id: str):
            raise ValueError("Patient not found")
        
        with pytest.raises(ValueError):
            view_patient(user_id="user123", patient_id="patient456")
        
        events = logger.query()
        assert len(events) == 1
        assert events[0]['success'] == False


# =============================================================================
# Token Manager Tests
# =============================================================================

class TestTokenManager:
    """Tests for TokenManager"""
    
    def test_generate_token(self):
        """Test token generation"""
        manager = TokenManager(secret_key="test_secret")
        
        token = manager.generate_token("user123")
        
        assert '.' in token
        assert len(token) > 50
    
    def test_validate_valid_token(self):
        """Test validating a valid token"""
        manager = TokenManager(secret_key="test_secret")
        
        token = manager.generate_token("user123", claims={'role': 'admin'})
        payload = manager.validate_token(token)
        
        assert payload is not None
        assert payload['sub'] == 'user123'
        assert payload['role'] == 'admin'
    
    def test_validate_invalid_token(self):
        """Test validating an invalid token"""
        manager = TokenManager(secret_key="test_secret")
        
        payload = manager.validate_token("invalid.token")
        
        assert payload is None
    
    def test_validate_tampered_token(self):
        """Test validating a tampered token"""
        manager = TokenManager(secret_key="test_secret")
        
        token = manager.generate_token("user123")
        
        # Tamper with token
        parts = token.split('.')
        tampered_token = parts[0] + ".tampered_signature"
        
        payload = manager.validate_token(tampered_token)
        
        assert payload is None
    
    def test_revoke_token(self):
        """Test token revocation"""
        manager = TokenManager(secret_key="test_secret")
        
        token = manager.generate_token("user123")
        
        # Token should be valid initially
        assert manager.validate_token(token) is not None
        
        # Revoke token
        manager.revoke_token(token)
        
        # Token should be invalid after revocation
        assert manager.validate_token(token) is None


# =============================================================================
# Permission Checker Tests
# =============================================================================

class TestPermissionChecker:
    """Tests for PermissionChecker"""
    
    def test_admin_has_all_permissions(self):
        """Test admin role has all permissions"""
        assert PermissionChecker.has_permission('admin', 'read') == True
        assert PermissionChecker.has_permission('admin', 'write') == True
        assert PermissionChecker.has_permission('admin', 'delete') == True
        assert PermissionChecker.has_permission('admin', 'admin') == True
    
    def test_viewer_limited_permissions(self):
        """Test viewer role has limited permissions"""
        assert PermissionChecker.has_permission('viewer', 'read') == True
        assert PermissionChecker.has_permission('viewer', 'write') == False
        assert PermissionChecker.has_permission('viewer', 'delete') == False
    
    def test_unknown_role(self):
        """Test unknown role has no permissions"""
        assert PermissionChecker.has_permission('unknown_role', 'read') == False
    
    def test_resource_specific_permissions(self):
        """Test resource-specific permissions"""
        assert PermissionChecker.has_permission('admin', 'read', 'patient_data') == True
        assert PermissionChecker.has_permission('admin', 'admin', 'patient_data') == False


class TestRequirePermissionDecorator:
    """Tests for require_permission decorator"""
    
    def test_allows_with_permission(self):
        """Test allows with correct permission"""
        @PermissionChecker.require_permission('read', 'patient_data')
        def view_data(user_role: str):
            return "data"
        
        result = view_data(user_role='admin')
        assert result == "data"
    
    def test_denies_without_permission(self):
        """Test denies without correct permission"""
        @PermissionChecker.require_permission('write', 'patient_data')
        def modify_data(user_role: str):
            return "modified"
        
        from core.error_handling import APIError
        with pytest.raises(APIError):
            modify_data(user_role='viewer')


# =============================================================================
# Data Encryption Tests
# =============================================================================

class TestDataEncryption:
    """Tests for DataEncryption"""
    
    def test_encrypt_decrypt_roundtrip(self):
        """Test encryption and decryption roundtrip"""
        encryptor = DataEncryption()
        
        original = "Sensitive patient data: SSN 123-45-6789"
        encrypted = encryptor.encrypt(original)
        decrypted = encryptor.decrypt(encrypted)
        
        assert decrypted == original
        assert encrypted != original
    
    def test_hash_pii(self):
        """Test PII hashing"""
        hash1 = DataEncryption.hash_pii("John Doe", salt="abc")
        hash2 = DataEncryption.hash_pii("John Doe", salt="abc")
        hash3 = DataEncryption.hash_pii("John Doe", salt="xyz")
        
        # Same input with same salt should produce same hash
        assert hash1 == hash2
        
        # Different salt should produce different hash
        assert hash1 != hash3
        
        # Hash should be fixed length
        assert len(hash1) == 64  # SHA-256 produces 64 hex characters


# =============================================================================
# Integration Tests
# =============================================================================

class TestSecurityIntegration:
    """Integration tests for security components"""
    
    def test_full_authentication_flow(self):
        """Test complete authentication flow"""
        manager = TokenManager(secret_key="test_secret")
        logger = AuditLogger()
        
        # User login
        token = manager.generate_token("user123", claims={'role': 'analyst'})
        
        logger.log(
            event_type=AuditEventType.LOGIN,
            user_id="user123",
            action="login",
            resource_type="system"
        )
        
        # Validate token and check permissions
        payload = manager.validate_token(token)
        assert payload is not None
        
        # Check permission
        assert PermissionChecker.has_permission('analyst', 'read') == True
        
        # Log access
        logger.log(
            event_type=AuditEventType.ACCESS,
            user_id="user123",
            action="view",
            resource_type="patient_data"
        )
        
        # Verify audit trail
        events = logger.query(user_id="user123")
        assert len(events) == 2
    
    def test_rate_limited_api_with_audit(self):
        """Test rate limited API with audit logging"""
        limiter = RateLimiter(RateLimitRule(requests_per_minute=5))
        logger = AuditLogger()
        
        for i in range(6):
            allowed, _ = limiter.is_allowed("user123")
            
            if allowed:
                logger.log(
                    event_type=AuditEventType.QUERY,
                    user_id="user123",
                    action="search",
                    resource_type="studies",
                    success=True
                )
            else:
                logger.log(
                    event_type=AuditEventType.SECURITY,
                    user_id="user123",
                    action="rate_limit_exceeded",
                    resource_type="api",
                    success=False
                )
        
        events = logger.query(event_type=AuditEventType.SECURITY)
        assert len(events) >= 1


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def audit_logger():
    """Provide fresh audit logger"""
    return AuditLogger()


@pytest.fixture
def rate_limiter():
    """Provide fresh rate limiter"""
    return RateLimiter()


@pytest.fixture
def token_manager():
    """Provide fresh token manager"""
    return TokenManager(secret_key="test_secret_key")


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
