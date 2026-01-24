"""Cache provider for analytics operations.

Provides thread-safe, TTL-based caching using cachetools.TTLCache.
Cache keys are deterministic across runs and processes.
"""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any, Callable

import cachetools


def args_hash(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Generate deterministic hash for function arguments.

    Uses JSON serialization with sorted keys to ensure consistent
    hash values across different Python processes and runs.

    Args:
        args: Positional arguments tuple.
        kwargs: Keyword arguments dictionary.

    Returns:
        SHA1 hex digest of the serialized arguments.
    """

    def serialize(obj: Any) -> Any:
        """Convert non-serializable objects to strings."""
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        if isinstance(obj, (list, tuple)):
            return [serialize(item) for item in obj]
        if isinstance(obj, dict):
            return {str(k): serialize(v) for k, v in sorted(obj.items())}
        return str(obj)

    serializable_args = serialize(args)
    serializable_kwargs = serialize(kwargs)

    payload = json.dumps(
        {"args": serializable_args, "kwargs": serializable_kwargs},
        sort_keys=True,
        ensure_ascii=True,
    )

    return hashlib.sha1(payload.encode("ascii")).hexdigest()


def cache_key(
    method_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[str, str]:
    """Generate cache key for a method call.

    Args:
        method_name: Name of the method being cached.
        args: Positional arguments tuple.
        kwargs: Keyword arguments dictionary.

    Returns:
        Tuple of (method_name, args_hash).
    """
    return (method_name, args_hash(args, kwargs))


class CacheProvider:
    """Thread-safe cache provider with TTL-based expiration.

    Wraps cachetools.TTLCache with threading.RLock for safe concurrent access.
    """

    def __init__(self, *, ttl: int | float = 600, maxsize: int = 1000):
        """Initialize cache provider.

        Args:
            ttl: Time-to-live for cache entries in seconds (default 600s).
            maxsize: Maximum number of items in cache (default 1000).
        """
        self._cache = cachetools.TTLCache(maxsize=maxsize, ttl=float(ttl))
        self._lock = threading.RLock()

    def get_or_set(self, key: tuple[str, str], factory: Callable[[], Any]) -> Any:
        """Get value from cache or compute using factory function.

        Thread-safe access with double-checked locking pattern.

        Args:
            key: Cache key tuple (method_name, args_hash).
            factory: Callable that produces the value if not cached.

        Returns:
            Cached or newly computed value.
        """
        with self._lock:
            if key in self._cache:
                return self._cache[key]

            value = factory()
            self._cache[key] = value
            return value

    def clear(self) -> None:
        """Clear all cached values."""
        with self._lock:
            self._cache.clear()

    def get(self, key: str) -> Any | None:
        """Get value from cache by string key.

        Args:
            key: String cache key.

        Returns:
            Cached value or None if not found or expired.
        """
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        """Set value in cache with string key.

        Args:
            key: String cache key.
            value: Value to cache.
        """
        with self._lock:
            self._cache[key] = value

    def get_stats(self) -> dict[str, int | float]:
        """Get cache statistics.

        Returns:
            Dictionary with current size and maxsize.
        """
        with self._lock:
            return {"current_size": len(self._cache), "max_size": self._cache.maxsize}
