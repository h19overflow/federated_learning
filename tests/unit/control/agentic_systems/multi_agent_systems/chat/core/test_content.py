"""
Tests for content processing utilities module.
"""

from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.core.content import (
    chunk_content,
    normalize_content,
)


class TestNormalizeContent:
    """Test normalize_content function."""

    def test_normalize_string_content(self):
        """Test normalizing string content."""
        content = "Hello world"
        result = normalize_content(content)

        assert result == "Hello world"
        assert isinstance(result, str)

    def test_normalize_none_content(self):
        """Test normalizing None content."""
        content = None
        result = normalize_content(content)

        assert result == ""
        assert isinstance(result, str)

    def test_normalize_list_of_strings(self):
        """Test normalizing list of strings."""
        content = ["Hello", " ", "world", "!"]
        result = normalize_content(content)

        assert result == "Hello world!"
        assert isinstance(result, str)

    def test_normalize_list_of_dicts_with_text(self):
        """Test normalizing list of dicts with text keys."""
        content = [
            {"text": "Hello"},
            {"text": " "},
            {"text": "world"},
            {"text": "!"},
        ]
        result = normalize_content(content)

        assert result == "Hello world!"
        assert isinstance(result, str)

    def test_normalize_mixed_list_content(self):
        """Test normalizing mixed list (strings and dicts)."""
        content = [
            {"text": "Hello"},
            " ",
            {"text": "world"},
            "!",
        ]
        result = normalize_content(content)

        assert result == "Hello world!"

    def test_normalize_empty_list(self):
        """Test normalizing empty list."""
        content = []
        result = normalize_content(content)

        assert result == ""

    def test_normalize_list_with_unexpected_types(self):
        """Test normalizing list with unexpected types (should log warning)."""
        content = ["Hello", 123, {"text": "world"}]
        result = normalize_content(content)

        # Should convert int to str
        assert "Hello" in result
        assert "123" in result
        assert "world" in result

    def test_normalize_unexpected_type(self):
        """Test normalizing unexpected type (should use str())."""
        content = 12345
        result = normalize_content(content)

        assert result == "12345"
        assert isinstance(result, str)

    def test_normalize_dict_without_text_key(self):
        """Test normalizing dict without text key."""
        content = {"other_key": "value"}
        result = normalize_content(content)

        assert isinstance(result, str)
        # Should convert dict to string representation

    def test_normalize_long_string(self):
        """Test normalizing long string."""
        content = "A" * 10000
        result = normalize_content(content)

        assert len(result) == 10000
        assert result == content

    def test_normalize_unicode_content(self):
        """Test normalizing unicode content."""
        content = "Hello 世界 🌍"
        result = normalize_content(content)

        assert result == "Hello 世界 🌍"

    def test_normalize_list_with_none_items(self):
        """Test normalizing list containing None items."""
        content = ["Hello", None, "world"]
        result = normalize_content(content)

        # None items should be converted to "None" string or handled
        assert isinstance(result, str)

    def test_normalize_empty_string(self):
        """Test normalizing empty string."""
        content = ""
        result = normalize_content(content)

        assert result == ""

    def test_normalize_whitespace(self):
        """Test normalizing whitespace."""
        content = ["   ", "Hello", "   ", "World", "   "]
        result = normalize_content(content)

        assert result == "   Hello   World   "

    def test_normalize_newlines(self):
        """Test normalizing newlines."""
        content = ["Line 1\n", "Line 2\n", "Line 3"]
        result = normalize_content(content)

        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result


class TestChunkContent:
    """Test chunk_content function."""

    def test_chunk_content_default_size(self):
        """Test chunking with default chunk size."""
        content = "Hello world, how are you today?"
        chunks = list(chunk_content(content))

        assert len(chunks) == 1
        assert chunks[0] == content

    def test_chunk_content_custom_size(self):
        """Test chunking with custom chunk size."""
        content = "Hello world, how are you today?"
        chunks = list(chunk_content(content, chunk_size=5))

        expected = ["Hello", " worl", "d, ho", "w are", " you ", "today", "?"]
        assert chunks == expected

    def test_chunk_content_exact_multiple(self):
        """Test chunking when content length is exact multiple of chunk size."""
        content = "HelloHelloHello"  # 15 chars
        chunks = list(chunk_content(content, chunk_size=5))

        expected = ["Hello", "Hello", "Hello"]
        assert chunks == expected

    def test_chunk_content_partial_last_chunk(self):
        """Test chunking with partial last chunk."""
        content = "HelloWorld"  # 10 chars
        chunks = list(chunk_content(content, chunk_size=3))

        expected = ["Hel", "loW", "orl", "d"]
        assert chunks == expected

    def test_chunk_content_empty_string(self):
        """Test chunking empty string."""
        content = ""
        chunks = list(chunk_content(content))

        assert len(chunks) == 0

    def test_chunk_content_single_chunk(self):
        """Test chunking content shorter than chunk size."""
        content = "Hi"
        chunks = list(chunk_content(content, chunk_size=50))

        assert len(chunks) == 1
        assert chunks[0] == "Hi"

    def test_chunk_content_large_chunk_size(self):
        """Test chunking with very large chunk size."""
        content = "Hello world"
        chunks = list(chunk_content(content, chunk_size=1000))

        assert len(chunks) == 1
        assert chunks[0] == content

    def test_chunk_content_size_one(self):
        """Test chunking with chunk size of 1."""
        content = "ABCD"
        chunks = list(chunk_content(content, chunk_size=1))

        expected = ["A", "B", "C", "D"]
        assert chunks == expected

    def test_chunk_content_preserves_content(self):
        """Test that chunking and recombining preserves original content."""
        content = "Hello world, this is a test string that should be preserved exactly."
        chunks = list(chunk_content(content, chunk_size=10))
        recombined = "".join(chunks)

        assert recombined == content

    def test_chunk_content_is_generator(self):
        """Test that chunk_content returns a generator."""
        content = "Hello world"
        result = chunk_content(content)

        # Should be a generator
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    def test_chunk_content_multiple_iterations(self):
        """Test that chunks can be iterated multiple times (consume and recreate)."""
        content = "ABCDEFGH"
        chunks1 = list(chunk_content(content, chunk_size=2))
        chunks2 = list(chunk_content(content, chunk_size=2))

        assert chunks1 == chunks2


class TestContentIntegration:
    """Integration tests for content processing."""

    def test_normalize_then_chunk(self):
        """Test normalizing then chunking content."""
        # Gemini-style content
        content = [
            {"text": "Hello "},
            {"text": "world"},
            " ",
            {"text": "this"},
            {"text": " is"},
            {"text": " a test"},
        ]

        normalized = normalize_content(content)
        chunks = list(chunk_content(normalized, chunk_size=5))

        assert len(chunks) > 1
        assert "".join(chunks) == "Hello world this is a test"

    def test_chunk_various_sizes(self):
        """Test chunking with various chunk sizes."""
        content = "The quick brown fox jumps over the lazy dog"

        chunk_sizes = [5, 10, 15, 20, 50]

        for size in chunk_sizes:
            chunks = list(chunk_content(content, chunk_size=size))
            recombined = "".join(chunks)
            assert recombined == content, f"Failed for chunk_size={size}"

    def test_normalize_complex_gemini_response(self):
        """Test normalizing complex Gemini response structure."""
        # Simulate complex Gemini response
        content = [
            {"text": "Based on the documents, "},
            {"text": "federated learning "},
            {"text": "is a "},
            "distributed ",
            {"text": "machine learning "},
            {"text": "approach."},
        ]

        result = normalize_content(content)
        expected = "Based on the documents, federated learning is a distributed machine learning approach."

        assert result == expected

    def test_edge_cases_together(self):
        """Test edge cases in combination."""
        # Test None, empty string, and normal content
        cases = [
            (None, ""),
            ("", ""),
            ("A", "A"),
            ("AB", "AB"),
        ]

        for input_content, expected_normalized in cases:
            normalized = normalize_content(input_content)
            assert normalized == expected_normalized

            if input_content:
                chunks = list(chunk_content(normalized))
                assert "".join(chunks) == normalized

    def test_chunk_with_special_characters(self):
        """Test chunking content with special characters."""
        content = "Hello\n\tWorld!@#$%^&*()_+-={}[]|\\:;\"'<>,.?/~`"
        chunks = list(chunk_content(content, chunk_size=5))

        recombined = "".join(chunks)
        assert recombined == content

    def test_normalize_list_with_dicts_missing_text(self):
        """Test normalizing list with dicts that don't have text key."""
        content = [
            {"text": "Hello"},
            {"other": "value"},
            {"text": "world"},
        ]
        result = normalize_content(content)

        # Should log warning for dict without text
        # but still process items that have text
        assert "Hello" in result
        assert "world" in result

    def test_long_content_processing(self):
        """Test processing very long content."""
        content = "A" * 100000  # 100k characters
        chunks = list(chunk_content(content, chunk_size=5000))

        recombined = "".join(chunks)
        assert recombined == content
        assert len(chunks) == 20  # 100000 / 5000
