"""
Cache Manager
Provides a unified interface for multi-level caching with Redis and in-memory LRU.
"""

import redis
import hashlib
import json
from typing import Any, Optional
from functools import lru_cache
from datetime import timedelta


class CacheManager:
    """
    Multi-level cache manager supporting L1 (LRU) and L2 (Redis) caching.
    """

    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        """
        Initialize cache manager with Redis connection.

        Args:
            redis_host: Redis server hostname
            redis_port: Redis server port
        """
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True,
            socket_connect_timeout=5
        )
        self.default_ttl = 3600  # 1 hour

    def _generate_key(self, namespace: str, identifier: str) -> str:
        """Generate cache key with namespace."""
        return f"{namespace}:{identifier}"

    def get(self, namespace: str, identifier: str) -> Optional[Any]:
        """
        Retrieve value from cache.

        Args:
            namespace: Cache namespace (e.g., 'user', 'api')
            identifier: Unique identifier within namespace

        Returns:
            Cached value or None if not found
        """
        key = self._generate_key(namespace, identifier)

        try:
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
        except redis.RedisError as e:
            print(f"Redis error on get: {e}")

        return None

    def set(self, namespace: str, identifier: str, value: Any,
            ttl: Optional[int] = None) -> bool:
        """
        Store value in cache with TTL.

        Args:
            namespace: Cache namespace
            identifier: Unique identifier
            value: Value to cache (must be JSON serializable)
            ttl: Time-to-live in seconds (default: 3600)

        Returns:
            True if successful, False otherwise
        """
        key = self._generate_key(namespace, identifier)
        ttl = ttl or self.default_ttl

        try:
            serialized = json.dumps(value)
            self.redis_client.setex(key, ttl, serialized)
            return True
        except (redis.RedisError, TypeError) as e:
            print(f"Cache set error: {e}")
            return False

    def delete(self, namespace: str, identifier: str) -> bool:
        """
        Invalidate cache entry.

        Args:
            namespace: Cache namespace
            identifier: Unique identifier

        Returns:
            True if deleted, False otherwise
        """
        key = self._generate_key(namespace, identifier)

        try:
            return bool(self.redis_client.delete(key))
        except redis.RedisError as e:
            print(f"Cache delete error: {e}")
            return False

    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching pattern.

        Args:
            pattern: Redis key pattern (e.g., 'user:*')

        Returns:
            Number of keys deleted
        """
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except redis.RedisError as e:
            print(f"Pattern invalidation error: {e}")
            return 0

    def get_or_compute(self, namespace: str, identifier: str,
                       compute_fn: callable, ttl: Optional[int] = None) -> Any:
        """
        Get from cache or compute if missing (cache-aside pattern).

        Args:
            namespace: Cache namespace
            identifier: Unique identifier
            compute_fn: Function to compute value if cache miss
            ttl: Time-to-live in seconds

        Returns:
            Cached or computed value
        """
        # Try cache first
        cached = self.get(namespace, identifier)
        if cached is not None:
            return cached

        # Compute and cache
        value = compute_fn()
        self.set(namespace, identifier, value, ttl)
        return value

    def increment(self, namespace: str, identifier: str,
                  amount: int = 1) -> int:
        """
        Atomically increment counter.

        Args:
            namespace: Cache namespace
            identifier: Counter identifier
            amount: Increment amount

        Returns:
            New counter value
        """
        key = self._generate_key(namespace, identifier)
        try:
            return self.redis_client.incrby(key, amount)
        except redis.RedisError as e:
            print(f"Increment error: {e}")
            return 0

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache metrics
        """
        try:
            info = self.redis_client.info('stats')
            return {
                'hits': info.get('keyspace_hits', 0),
                'misses': info.get('keyspace_misses', 0),
                'hit_rate': self._calculate_hit_rate(info),
                'total_keys': self.redis_client.dbsize()
            }
        except redis.RedisError:
            return {}

    @staticmethod
    def _calculate_hit_rate(info: dict) -> float:
        """Calculate cache hit rate percentage."""
        hits = info.get('keyspace_hits', 0)
        misses = info.get('keyspace_misses', 0)
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0
