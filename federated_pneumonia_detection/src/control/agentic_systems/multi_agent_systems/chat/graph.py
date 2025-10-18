from langgraph.pregel.main import asyncio
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.nodes.fetch_context import fetch_context
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.nodes.answer import answer
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.chat.schemas import ChatState
from langgraph.graph import StateGraph, START, END


def build_graph():
    graph = StateGraph(ChatState)
    graph.add_node("fetch_context", fetch_context)
    graph.add_node("answer", answer)
    graph.add_edge(START, "fetch_context")
    graph.add_edge("fetch_context", "answer")
    graph.add_edge("answer", END)
    return graph.compile()

async def async_invoke(graph, query: str):
    return await graph.ainvoke(ChatState(query=query))
if __name__ == "__main__":
    graph = build_graph()
    result = asyncio.run(async_invoke(graph, "I have a question about federated learning for pneumonia detection, and what are the main components in a federated learning system?"))
    print(result)