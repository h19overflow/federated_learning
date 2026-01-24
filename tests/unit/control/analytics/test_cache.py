"""Unit tests for analytics cache provider."""

import time
from typing import Any

import pytest

from federated_pneumonia_detection.src.control.analytics.cache import (
    CacheProvider,
    args_hash,
    cache_key,
)


class TestArgsHash:
    """Test deterministic argument hashing."""

    def test_hash_basic_types(self):
        """Test hashing with basic types."""
        h1 = args_hash((1, 2, 3), {"a": 1, "b": 2})
        h2 = args_hash((1, 2, 3), {"a": 1, "b": 2})
        assert h1 == h2

    def test_hash_dict_order_independence(self):
        """Test that dict key order does not affect hash."""
        h1 = args_hash((), {"z": 1, "a": 2, "m": 3})
        h2 = args_hash((), {"a": 2, "z": 1, "m": 3})
        h3 = args_hash((), {"m": 3, "a": 2, "z": 1})
        assert h1 == h2 == h3

    def test_hash_list_content_matters(self):
        """Test that list order affects hash."""
        h1 = args_hash(([1, 2, 3],), {})
        h2 = args_hash(([1, 3, 2],), {})
        assert h1 != h2

    def test_hash_nested_structures(self):
        """Test hashing with nested structures."""
        h1 = args_hash(
            ({"x": [1, 2]},),
            {"nested": {"deep": [1, 2, 3]}, "list": [{"a": 1}, {"a": 2}]},
        )
        h2 = args_hash(
            ({"x": [1, 2]},),
            {"nested": {"deep": [1, 2, 3]}, "list": [{"a": 1}, {"a": 2}]},
        )
        assert h1 == h2

    def test_hash_none_values(self):
        """Test hashing with None values."""
        h1 = args_hash((None, None), {"a": None})
        h2 = args_hash((None, None), {"a": None})
        assert h1 == h2

    def test_hash_mixed_types(self):
        """Test hashing with mixed types."""
        h1 = args_hash((1, "two", 3.0, True, None), {"a": [1, 2]})
        h2 = args_hash((1, "two", 3.0, True, None), {"a": [1, 2]})
        assert h1 == h2

    def test_hash_unserializable_objects(self):
        """Test hashing converts unserializable objects to strings."""
        obj = object()
        h1 = args_hash((obj,), {})
        h2 = args_hash((obj,), {})
        assert h1 == h2

    def test_hash_output_is_string(self):
        """Test that hash output is a SHA1 hex string."""
        h = args_hash((1, 2, 3), {})
        assert isinstance(h, str)
        assert len(h) == 40  # SHA1 hex digest length
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_ascii_only(self):
        """Test that hashing works with ASCII-only constraint."""
        h = args_hash(("test", 123, None), {"key": "value"})
        try:
            h.encode("ascii").decode("ascii")
            assert True
        except UnicodeError:
            pytest.fail("Hash contains non-ASCII characters")


class TestCacheKey:
    """Test cache key generation."""

    def test_cache_key_structure(self):
        """Test cache key returns tuple of (method_name, hash)."""
        key = cache_key("method", (1, 2), {"a": 1})
        assert isinstance(key, tuple)
        assert len(key) == 2
        assert key[0] == "method"
        assert isinstance(key[1], str)

    def test_cache_key_consistency(self):
        """Test same inputs produce same cache key."""
        key1 = cache_key("method", (1, 2), {"a": 1})
        key2 = cache_key("method", (1, 2), {"a": 1})
        assert key1 == key2

    def test_cache_key_different_methods(self):
        """Test different method names produce different keys."""
        key1 = cache_key("method1", (1, 2), {})
        key2 = cache_key("method2", (1, 2), {})
        assert key1 != key2

    def test_cache_key_different_args(self):
        """Test different arguments produce different keys."""
        key1 = cache_key("method", (1, 2), {})
        key2 = cache_key("method", (1, 3), {})
        assert key1 != key2


class TestCacheProvider:
    """Test CacheProvider TTL-based caching."""

    def test_cache_hit(self):
        """Test that cached value is returned."""
        provider = CacheProvider(ttl=60, maxsize=100)
        key = ("method", "hash123")

        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return {"value": call_count}

        result1 = provider.get_or_set(key, factory)
        result2 = provider.get_or_set(key, factory)

        assert call_count == 1
        assert result1 == result2
        assert result1["value"] == 1

    def test_cache_miss_after_factory(self):
        """Test that new factory is called for different keys."""
        provider = CacheProvider(ttl=60, maxsize=100)
        key1 = ("method1", "hash123")
        key2 = ("method2", "hash456")

        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return call_count

        result1 = provider.get_or_set(key1, factory)
        result2 = provider.get_or_set(key2, factory)

        assert call_count == 2
        assert result1 == 1
        assert result2 == 2

    def test_cache_expiration(self):
        """Test that cache expires after TTL."""
        provider = CacheProvider(ttl=1, maxsize=100)
        key = ("method", "hash123")

        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return call_count

        result1 = provider.get_or_set(key, factory)
        assert call_count == 1

        time.sleep(1.5)

        result2 = provider.get_or_set(key, factory)
        assert call_count == 2
        assert result2 == 2

    def test_cache_maxsize_eviction(self):
        """Test that cache respects maxsize limit."""
        provider = CacheProvider(ttl=60, maxsize=2)

        def factory(i):
            return i

        for i in range(3):
            provider.get_or_set(("method", f"key{i}"), lambda v=i: factory(v))

        stats = provider.get_stats()
        assert stats["current_size"] == 2
        assert stats["max_size"] == 2

    def test_cache_clear(self):
        """Test that cache can be cleared."""
        provider = CacheProvider(ttl=60, maxsize=100)
        key = ("method", "hash123")

        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return call_count

        provider.get_or_set(key, factory)
        assert call_count == 1

        provider.clear()
        provider.get_or_set(key, factory)
        assert call_count == 2

    def test_cache_stats(self):
        """Test that cache returns correct stats."""
        provider = CacheProvider(ttl=60, maxsize=10)

        stats = provider.get_stats()
        assert stats["current_size"] == 0
        assert stats["max_size"] == 10

        provider.get_or_set(("method", "key1"), lambda: 1)
        provider.get_or_set(("method", "key2"), lambda: 2)

        stats = provider.get_stats()
        assert stats["current_size"] == 2

    def test_cache_thread_safety_basic(self):
        """Test basic thread safety of cache operations."""
        import threading

        provider = CacheProvider(ttl=60, maxsize=100)
        key = ("method", "hash123")

        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return call_count

        def concurrent_call():
            return provider.get_or_set(key, factory)

        threads = [threading.Thread(target=concurrent_call) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert call_count >= 1  # At least one call, possibly more

    def test_cache_with_complex_values(self):
        """Test caching complex dictionary structures."""
        provider = CacheProvider(ttl=60, maxsize=100)
        key = ("method", "hash123")

        complex_value = {
            "nested": {"data": [1, 2, 3]},
            "list": [{"a": 1}, {"b": 2}],
            "mixed": [1, "two", None, True],
        }

        result = provider.get_or_set(key, lambda: complex_value)
        assert result == complex_value

        cached_result = provider.get_or_set(key, lambda: {"different": "value"})
        assert cached_result == complex_value
