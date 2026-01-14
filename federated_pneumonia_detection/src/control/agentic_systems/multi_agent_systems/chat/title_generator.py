"""
Chat Session Title Generator
=============================

Generates concise 3-4 word titles from user queries using Gemini 2.0 Flash.

Usage:
    title = generate_chat_title("What is federated learning?")
    # Returns: "Understanding Federated Learning"
"""

import logging

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, field_validator

load_dotenv()
logger = logging.getLogger(__name__)


# Pydantic model for structured output
class ChatTitle(BaseModel):
    """Structured chat title with validation constraints."""

    title: str = Field(
        description="A concise 3-4 word title in title case, no punctuation",
        max_length=60,  # Character limit to ensure brevity
    )

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate and normalize title format."""
        # Remove quotes and extra punctuation
        v = v.replace('"', "").replace("'", "").strip()
        v = "".join(char for char in v if char.isalnum() or char.isspace())

        # Enforce max 4 words
        words = v.split()
        if len(words) > 4:
            v = " ".join(words[:4])

        # Apply title case
        v = v.title()

        return v


# Singleton agent instance for performance
_title_agent = None


def _get_title_agent():
    """Get or create the title generation agent with structured output."""
    global _title_agent
    if _title_agent is None:
        base_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.3,  # Low temp for consistent titles
            max_tokens=20,  # Titles are short
        )
        _title_agent = create_agent(
            model=base_model,
            tools=[],  # No tools needed for title generation
            response_format=ChatTitle,
        )
    return _title_agent


def generate_chat_title(query: str) -> str:
    """
    Generate a concise 3-4 word title from a user query.

    Args:
        query: User's first message in the chat session

    Returns:
        A 3-4 word title, or fallback to first 3 words if generation fails
    """
    try:
        logger.info(f"[TitleGen] Generating title for query: '{query[:50]}...'")

        agent = _get_title_agent()

        prompt = f"""Summarize the following query into exactly 3-4 words for a chat session title.
Rules:
- Use title case
- Be descriptive and specific
- No punctuation
- Maximum 4 words

Query: {query}

Title:"""

        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})

        # Extract structured response from agent result
        response: ChatTitle = result.get("structured_response")
        logger.debug(f"[TitleGen] Structured response: title='{response.title}'")

        logger.info(f"[TitleGen] Generated title: '{response.title}'")
        return response.title

    except Exception as e:
        logger.warning(f"[TitleGen] Failed to generate title: {e}")
        # Fallback: first 3 words of query, capitalized
        words = query.split()[:3]
        fallback = " ".join(word.capitalize() for word in words)
        logger.info(f"[TitleGen] Using fallback title: '{fallback}'")
        return fallback
