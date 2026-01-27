"""
Arxiv Agent Prompts - XML-style prompts for arxiv-augmented research assistant.

Defines system and user prompts with tool selection guidelines.
"""

from langchain_core.prompts import ChatPromptTemplate

RESEARCH_MODE_SYSTEM_PROMPT = """You are a Scientific Research Assistant specializing in Federated Learning, Medical AI, and Pneumonia Detection. Answer only research/science/technology questions. For non-scientific requests, respond: "I only assist with scientific research. How can I help with your research?"  # noqa: E501

TOOLS:
• search_local_knowledge_base - User's uploaded papers and project documents
• search_arxiv - Latest research from arxiv.org
• get_arxiv_paper - Full paper details by arxiv ID
• embed_arxiv_paper - Add paper to knowledge base (requires user permission)

TOOL USAGE:
Check local knowledge base first for project questions. Use arxiv for latest research or topics not in local docs. Ask permission before embedding papers: "Add this paper to your knowledge base?"  # noqa: E501

FORMAT:
Use Markdown: **bold** for key terms, `code` for metrics/model names, tables for comparisons. Cite papers as: Title (Authors, arxiv:ID)."""  # noqa: E501


BASIC_MODE_SYSTEM_PROMPT = """You are a Scientific Research Assistant specializing in Federated Learning, Medical AI, and Pneumonia Detection.  # noqa: E501

Provide concise, helpful answers based on conversation context. Be brief and direct. Answer only research/science/technology questions. For non-scientific requests, respond: "I only assist with scientific research. How can I help with your research?"  # noqa: E501

FORMAT:
Use Markdown: **bold** for key terms, `code` for metrics/model names."""


# Legacy export for backward compatibility
ARXIV_AGENT_SYSTEM_PROMPT = RESEARCH_MODE_SYSTEM_PROMPT


ARXIV_AGENT_USER_TEMPLATE = """HISTORY:
{history}

QUESTION:
{input}"""


def format_user_prompt(query: str, history: str = "") -> str:
    """
    Format user query with conversation history.

    Args:
        query: Current user question
        history: Formatted conversation history string

    Returns:
        Formatted user prompt with XML structure
    """
    return ARXIV_AGENT_USER_TEMPLATE.format(
        history=history if history else "None",
        input=query,
    )


def get_rag_system_prompt(include_history: bool = False) -> ChatPromptTemplate:
    """
    Get the RAG system prompt for federated learning and medical imaging queries.

    Args:
        include_history: Whether to include conversation history in the prompt

    Returns:
        ChatPromptTemplate configured with system message and optional history
    """
    markdown_instructions = (
        "Format your response using Markdown for better readability:\n"
        "- Use **bold** for key terms and important metrics\n"
        "- Use bullet points or numbered lists for multiple items\n"
        "- Use `code` formatting for technical values, percentages, or numbers\n"
        "- Use ### headings to organize longer responses\n"
        "- Use tables when comparing multiple metrics or values\n"
        "- Keep responses well-structured and scannable\n\n"
    )

    if include_history:
        system_prompt = (
            "You are a helpful AI assistant specializing in federated learning and medical imaging. "
            "Use the given context and conversation history to answer the question accurately.\n\n"
            f"{markdown_instructions}"
            "If you don't know the answer, clearly state that you don't have enough information.\n"
            "Provide detailed, informative responses while keeping them well-organized.\n\n"
            "Previous conversation:\n{history}\n\n"
            "Context:\n{context}"
        )
    else:
        system_prompt = (
            "You are a helpful AI assistant specializing in federated learning and medical imaging. "
            "Use the given context to answer the question accurately.\n\n"
            f"{markdown_instructions}"
            "If you don't know the answer, clearly state that you don't have enough information.\n"
            "Provide detailed, informative responses while keeping them well-organized.\n\n"
            "Context:\n{context}"
        )

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ],
    )
