"""
Query Router - Classifies queries as basic conversation or tool-augmented research.

Lightweight classification using Gemini 2.0 Flash with structured output to determine
whether a query requires tools (RAG, arxiv, MCP) or can be answered conversationally.

Usage:
    mode = classify_query("What papers discuss federated learning?")
    # Returns: "research"

    mode = classify_query("Thanks for the explanation!")
    # Returns: "basic"
"""

import logging
from typing import Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# Pydantic model for structured output
class QueryClassification(BaseModel):
    """Structured classification response with constrained mode field."""

    mode: Literal["research", "basic"] = Field(
        description="Query mode: 'research' requires tools (search, retrieval), 'basic' is conversational"
    )


# Singleton LLM instance for performance
_router_llm = None


def _get_router_llm():
    """Get or create the router classification LLM with structured output."""
    global _router_llm
    if _router_llm is None:
        base_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.0,  # Deterministic classification
            max_tokens=50,  # Small response
        )
        # Use with_structured_output for simple classification (no agent needed)
        _router_llm = base_model.with_structured_output(QueryClassification)
    return _router_llm


ROUTER_CLASSIFICATION_PROMPT = """Classify this query as either "research" or "basic".

RESEARCH queries need tools (search, retrieval, analysis):
- Questions about papers, research topics, technical details
- Requests for comparisons, citations, specific studies
- Questions requiring external knowledge or sources
- Examples: "What papers discuss federated learning?", "Compare ResNet and VGG", "Find papers by Andrew Ng"

BASIC queries can be answered conversationally:
- Greetings, thanks, acknowledgments
- Clarifications of previous responses
- Simple explanations without requiring sources
- Follow-up questions about already-retrieved information
- Examples: "Hello", "Thanks!", "Can you explain that more?", "What did you mean by recall?"

Query: {query}

Classification:"""


def classify_query(query: str) -> str:
    """
    Classify query as requiring tool-augmented research or basic conversation.

    Args:
        query: User's input query

    Returns:
        "research" if tools needed, "basic" if conversational response sufficient.
        Defaults to "research" on failure (safer fallback).
    """
    try:
        logger.info(f"[QueryRouter] Classifying query: '{query[:50]}...'")

        llm = _get_router_llm()
        prompt = ROUTER_CLASSIFICATION_PROMPT.format(query=query)

        logger.debug("[QueryRouter] Invoking LLM with structured output...")
        response: QueryClassification = llm.invoke(prompt)

        logger.info(f"[QueryRouter] Classification: {response.mode}")
        return response.mode

    except Exception as e:
        logger.error(f"[QueryRouter] Classification failed: {e}", exc_info=True)
        logger.info("[QueryRouter] Defaulting to research mode due to error")
        return "research"  # Safe fallback - always allow tools if uncertain
