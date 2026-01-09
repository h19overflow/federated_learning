"""
Query Router - Classifies queries as basic conversation or tool-augmented research.

Lightweight classification agent using Gemini 2.0 Flash to determine whether
a query requires tools (RAG, arxiv, MCP) or can be answered conversationally.

Usage:
    mode = classify_query("What papers discuss federated learning?")
    # Returns: "research"

    mode = classify_query("Thanks for the explanation!")
    # Returns: "basic"
"""

import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Singleton model instance for performance
_router_model = None


def _get_router_model() -> ChatGoogleGenerativeAI:
    """Get or create the router classification model instance."""
    global _router_model
    if _router_model is None:
        _router_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.0,  # Deterministic classification
            max_tokens=10,    # Single word response
        )
    return _router_model


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

Classification (respond with only "research" or "basic"):"""


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

        model = _get_router_model()
        prompt = ROUTER_CLASSIFICATION_PROMPT.format(query=query)

        logger.debug(f"[QueryRouter] Invoking classification model...")
        response = model.invoke([HumanMessage(content=prompt)])
        logger.debug(f"[QueryRouter] Raw response: '{response.content}'")

        classification = response.content.strip().lower()

        # Validate classification
        if classification not in ["research", "basic"]:
            logger.warning(
                f"[QueryRouter] Invalid classification '{classification}', defaulting to research"
            )
            return "research"

        logger.info(f"[QueryRouter] Classification: {classification}")
        return classification

    except Exception as e:
        logger.error(f"[QueryRouter] Classification failed: {e}", exc_info=True)
        logger.info("[QueryRouter] Defaulting to research mode due to error")
        return "research"  # Safe fallback - always allow tools if uncertain
