"""
Response Caching System for Neural Clinical Data Mesh
Implements 30-minute TTL caching for API responses
"""

import time
import hashlib
import json
from typing import Dict, Any, Optional
from threading import Lock
from datetime import datetime, timedelta

class ResponseCache:
    """
    Thread-safe response caching with TTL support
    """

    def __init__(self, ttl_seconds: int = 1800):  # 30 minutes default
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.lock = Lock()

    def _generate_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate cache key from endpoint and parameters"""
        # Sort params for consistent key generation
        sorted_params = json.dumps(params, sort_keys=True)
        key_string = f"{endpoint}:{sorted_params}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, endpoint: str, params: Dict[str, Any]) -> Optional[Any]:
        """Get cached response if valid"""
        key = self._generate_key(endpoint, params)

        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if time.time() - entry['timestamp'] < self.ttl_seconds:
                    print(f"✅ Cache hit for {endpoint}")
                    return entry['data']
                else:
                    # Expired, remove it
                    del self.cache[key]
                    print(f"⏰ Cache expired for {endpoint}")

        print(f"❌ Cache miss for {endpoint}")
        return None

    def set(self, endpoint: str, params: Dict[str, Any], data: Any) -> None:
        """Cache response data"""
        key = self._generate_key(endpoint, params)

        with self.lock:
            self.cache[key] = {
                'data': data,
                'timestamp': time.time(),
                'endpoint': endpoint,
                'params': params
            }

        print(f"💾 Cached response for {endpoint}")

    def invalidate_endpoint(self, endpoint: str) -> int:
        """Invalidate all cached responses for an endpoint"""
        invalidated = 0
        with self.lock:
            keys_to_remove = []
            for key, entry in self.cache.items():
                if entry['endpoint'] == endpoint:
                    keys_to_remove.append(key)
                    invalidated += 1

            for key in keys_to_remove:
                del self.cache[key]

        if invalidated > 0:
            print(f"🗑️  Invalidated {invalidated} cached responses for {endpoint}")

        return invalidated

    def invalidate_all(self) -> int:
        """Invalidate all cached responses"""
        with self.lock:
            count = len(self.cache)
            self.cache.clear()

        print(f"🗑️  Invalidated all {count} cached responses")
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_entries = len(self.cache)
            if total_entries == 0:
                return {
                    'total_entries': 0,
                    'oldest_entry': None,
                    'newest_entry': None,
                    'avg_age_seconds': 0
                }

            now = time.time()
            ages = [now - entry['timestamp'] for entry in self.cache.values()]

            return {
                'total_entries': total_entries,
                'oldest_entry_seconds': max(ages),
                'newest_entry_seconds': min(ages),
                'avg_age_seconds': sum(ages) / len(ages),
                'ttl_seconds': self.ttl_seconds
            }

    def cleanup_expired(self) -> int:
        """Remove expired entries"""
        now = time.time()
        expired_keys = []

        with self.lock:
            for key, entry in self.cache.items():
                if now - entry['timestamp'] >= self.ttl_seconds:
                    expired_keys.append(key)

            for key in expired_keys:
                del self.cache[key]

        if expired_keys:
            print(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")

        return len(expired_keys)

# Global cache instance
response_cache = ResponseCache(ttl_seconds=1800)  # 30 minutes

def cached_endpoint(endpoint_name: str):
    """
    Decorator for caching Flask endpoints
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get request parameters (simplified - in real implementation,
            # would extract from Flask request object)
            params = {}

            # Try to get from cache
            cached_data = response_cache.get(endpoint_name, params)
            if cached_data is not None:
                return cached_data

            # Execute function
            result = func(*args, **kwargs)

            # Cache result
            response_cache.set(endpoint_name, params, result)

            return result
        return wrapper
    return decorator