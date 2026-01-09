"""
Chat Session Title Generator
=============================

Generates concise 3-4 word titles from user queries using Gemini 2.0 Flash.

Usage:
    title = generate_chat_title("What is federated learning?")
    # Returns: "Understanding Federated Learning"
"""

import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Singleton model instance for performance
_title_model = None


def _get_title_model() -> ChatGoogleGenerativeAI:
    """Get or create the title generation model instance."""
    global _title_model
    if _title_model is None:
        _title_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.3,  # Low temp for consistent titles
            max_tokens=20,    # Titles are short
        )
    return _title_model


def generate_chat_title(query: str) -> str:
    """
    Generate a concise 3-4 word title from a user query.

    Args:
        query: User's first message in the chat session

    Returns:
        A 3-4 word title, or fallback to first 3 words if generation fails

    Examples:
        >>> generate_chat_title("What is federated learning?")
        "Understanding Federated Learning"

        >>> generate_chat_title("How do I train a model?")
        "Model Training Guide"
    """
    try:
        logger.info(f"[TitleGen] Generating title for query: '{query[:50]}...'")

        model = _get_title_model()

        prompt = f"""Summarize the following query into exactly 3-4 words for a chat session title.
Rules:
- Use title case
- Be descriptive and specific
- No punctuation
- Maximum 4 words

Query: {query}

Title:"""

        response = model.invoke([HumanMessage(content=prompt)])
        title = response.content.strip()

        # Clean up the title
        title = title.replace('"', '').replace("'", '').strip()

        # Validate length (max 4 words)
        words = title.split()
        if len(words) > 4:
            title = ' '.join(words[:4])

        logger.info(f"[TitleGen] Generated title: '{title}'")
        return title

    except Exception as e:
        logger.warning(f"[TitleGen] Failed to generate title: {e}")
        # Fallback: first 3 words of query, capitalized
        words = query.split()[:3]
        fallback = ' '.join(word.capitalize() for word in words)
        logger.info(f"[TitleGen] Using fallback title: '{fallback}'")
        return fallback
