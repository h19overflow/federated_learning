"""Utility functions for chat endpoint context building and enhancement."""

import json
import logging
from typing import Optional

from sqlalchemy import func

logger = logging.getLogger(__name__)


def sse_pack(data: dict) -> str:
    """Pack a dictionary into Server-Sent Events format."""
    return f"data: {json.dumps(data)}\n\n"


def sse_error(message: str, error_type: str = "error") -> str:
    """Create a formatted SSE error message."""
    return sse_pack({"type": error_type, "message": message})


def ensure_db_session(session_id: str, query: str) -> None:
    """Ensure a chat session exists in the database."""
    from federated_pneumonia_detection.src.boundary.CRUD.chat_history import (
        create_chat_session,
        get_chat_session,
    )

    try:
        existing_session = get_chat_session(session_id)
        if not existing_session:
            logger.info(f"[Helper] Creating new DB session: {session_id}")
            create_chat_session(title=query[:50] + "...", session_id=session_id)
    except Exception as e:
        logger.warning(f"[Helper] Failed to ensure DB session (non-fatal): {e}")


async def prepare_enhanced_query(query: str, run_id: Optional[int]) -> str:
    """Enhance a query with run context if a run_id is provided (async)."""
    if run_id is None:
        return query

    try:
        logger.info(f"[Helper] Enhancing query with run context for run_id: {run_id}")
        enhanced = await enhance_query_with_run_context(query, run_id)
        logger.info("[Helper] Query enhanced successfully")
        return enhanced
    except Exception as e:
        logger.error(f"[Helper] Failed to enhance query (using original): {e}")
        return query


def build_run_context(
    db,
    run_id: int,
    run_crud,
    run_metric_crud,
    server_evaluation_crud,
) -> Optional[str]:
    """
    Build comprehensive context string for a training run.

    Args:
        db: Database session
        run_id: ID of the training run
        run_crud: CRUD operations for runs
        run_metric_crud: CRUD operations for run metrics
        server_evaluation_crud: CRUD operations for server evaluations

    Returns:
        Formatted context string or None if run not found
    """
    try:
        # Fetch run with all related data
        run_data = run_crud.get(db, run_id)

        if not run_data:
            logger.warning(f"Run with id={run_id} not found")
            return None

        logger.info(f"[ChatContext] Fetching comprehensive data for run_id={run_id}")

        # Build comprehensive context string
        context_info = f"\n\n[TRAINING RUN CONTEXT - Run #{run_id}]\n"
        context_info += "=" * 60 + "\n"

        # Add basic run information
        context_info += _build_basic_run_info(run_data)

        # Get aggregated metrics information (avoid loading all into memory)
        metrics_summary = (
            db.query(
                run_metric_crud.model.dataset_type,
                run_metric_crud.model.metric_name,
                func.count(run_metric_crud.model.id).label("count"),
                func.max(run_metric_crud.model.metric_value).label("best"),
                func.min(run_metric_crud.model.metric_value).label("worst"),
                func.avg(run_metric_crud.model.metric_value).label("average"),
            )
            .filter(run_metric_crud.model.run_id == run_id)
            .group_by(
                run_metric_crud.model.dataset_type,
                run_metric_crud.model.metric_name,
            )
            .all()
        )

        if metrics_summary:
            context_info += _build_metrics_summary_from_aggregation(metrics_summary)

        # Add federated-specific information if applicable
        if run_data.training_mode == "federated":
            context_info += _build_federated_details(
                run_data,
                server_evaluation_crud,
                db,
                run_id,
            )

        # Add instructions for AI
        context_info += _build_ai_instructions()

        logger.info(f"[OK] Built comprehensive run context (run_id={run_id})")
        logger.info(f"Context length: {len(context_info)} characters")

        return context_info

    except Exception as e:
        logger.error(
            f"[ERROR] Error building run context for run_id={run_id}: {e}",
            exc_info=True,
        )
        return None


def _build_basic_run_info(run_data) -> str:
    """Build basic run information section."""
    info = f"Training Mode: {run_data.training_mode}\n"
    info += f"Status: {run_data.status}\n"
    info += f"Start Time: {run_data.start_time}\n"

    if run_data.end_time:
        info += f"End Time: {run_data.end_time}\n"
        duration = run_data.end_time - run_data.start_time
        info += f"Duration: {duration}\n"

    if run_data.run_description:
        info += f"Description: {run_data.run_description}\n"

    if run_data.source_path:
        info += f"Source Path: {run_data.source_path}\n"

    info += "\n"
    return info


def _build_metrics_summary(all_metrics) -> str:
    """Build metrics summary section."""
    summary = f"METRICS SUMMARY ({len(all_metrics)} total metrics recorded):\n"
    summary += "-" * 60 + "\n"

    # Group metrics by type
    metrics_by_type = {}
    for metric in all_metrics:
        key = f"{metric.dataset_type}_{metric.metric_name}"
        if key not in metrics_by_type:
            metrics_by_type[key] = []
        metrics_by_type[key].append(
            {
                "step": metric.step,
                "value": metric.metric_value,
                "client_id": metric.client_id,
                "round_id": metric.round_id,
            },
        )

    # Report on each metric type
    for metric_key, values in metrics_by_type.items():
        values_list = [v["value"] for v in values]
        summary += f"\n{metric_key}:\n"
        summary += f"  - Count: {len(values)}\n"
        summary += f"  - Best: {max(values_list):.4f}\n"
        summary += f"  - Worst: {min(values_list):.4f}\n"
        summary += f"  - Latest: {values_list[-1]:.4f}\n"
        summary += f"  - Average: {sum(values_list) / len(values_list):.4f}\n"

    return summary


def _build_metrics_summary_from_aggregation(metrics_summary) -> str:
    """Build metrics summary from SQL aggregation results."""
    summary = f"METRICS SUMMARY ({sum(m.count for m in metrics_summary)} total metrics recorded):\n"
    summary += "-" * 60 + "\n"

    for metric in metrics_summary:
        metric_key = f"{metric.dataset_type}_{metric.metric_name}"
        summary += f"\n{metric_key}:\n"
        summary += f"  - Count: {metric.count}\n"
        summary += f"  - Best: {metric.best:.4f}\n"
        summary += f"  - Worst: {metric.worst:.4f}\n"
        summary += f"  - Average: {metric.average:.4f}\n"

    return summary


def _build_federated_details(
    run_data,
    server_evaluation_crud,
    db,
    run_id: int,
) -> str:
    """Build federated learning specific details section."""
    details = "\n" + "=" * 60 + "\n"
    details += "FEDERATED LEARNING DETAILS:\n"
    details += "-" * 60 + "\n"

    # Client information
    if run_data.clients:
        details += f"Number of Clients: {len(run_data.clients)}\n"

    # Server evaluations
    server_evals = server_evaluation_crud.get_by_run(db, run_id)
    if server_evals:
        details += f"\nServer Evaluations: {len(server_evals)} rounds\n"
        details += "-" * 60 + "\n"

        for eval in server_evals:
            details += f"\nRound {eval.round_number}:\n"
            details += f"  - Loss: {eval.loss:.4f}\n"
            if eval.accuracy is not None:
                details += f"  - Accuracy: {eval.accuracy:.4f}\n"
            if eval.precision is not None:
                details += f"  - Precision: {eval.precision:.4f}\n"
            if eval.recall is not None:
                details += f"  - Recall: {eval.recall:.4f}\n"
            if eval.f1_score is not None:
                details += f"  - F1 Score: {eval.f1_score:.4f}\n"
            if eval.auroc is not None:
                details += f"  - AUROC: {eval.auroc:.4f}\n"
            if eval.num_samples:
                details += f"  - Samples: {eval.num_samples}\n"

    return details


def _build_ai_instructions() -> str:
    """Build AI instructions section."""
    instructions = "\n" + "=" * 60 + "\n"
    instructions += "INSTRUCTIONS FOR AI:\n"
    instructions += (
        "Use the above detailed information to answer the user's question.\n"
    )
    instructions += (
        "Never mention the status of the run, just answer based on the data provided.\n"
    )
    instructions += (
        "Provide specific numbers, trends, and insights based on this data.\n"
    )
    instructions += (
        "If the user asks about comparisons, note what data is available.\n\n"
    )
    instructions += "FORMAT YOUR RESPONSE USING MARKDOWN:\n"
    instructions += "- Use **bold** for key metrics and important values\n"
    instructions += "- Use bullet points or numbered lists for multiple items\n"
    instructions += (
        "- Use `code` formatting for percentages, numbers, and metric names\n"
    )
    instructions += "- Use tables when comparing metrics across rounds or clients\n"
    instructions += "- Use ### headings to organize sections in longer responses\n"
    instructions += "=" * 60 + "\n\n"
    return instructions


async def enhance_query_with_run_context(query: str, run_id: int) -> str:
    """
    Enhance a query with training run context (async).

    Args:
        query: Original user query
        run_id: ID of the training run to fetch context for

    Returns:
        Enhanced query string with context appended
    """
    import asyncio
    from federated_pneumonia_detection.src.boundary.CRUD import (
        run_crud,
        run_metric_crud,
        server_evaluation_crud,
    )
    from federated_pneumonia_detection.src.boundary.engine import get_session

    def _build_context_sync():
        """Synchronous DB operation to run in executor."""
        db = get_session()
        try:
            context_info = build_run_context(
                db,
                run_id,
                run_crud,
                run_metric_crud,
                server_evaluation_crud,
            )
            return context_info
        finally:
            if db:
                db.close()

    # Run blocking DB operation in thread pool to avoid blocking event loop
    loop = asyncio.get_event_loop()
    context_info = await loop.run_in_executor(None, _build_context_sync)

    if context_info:
        return query + context_info
    else:
        logger.warning(
            f"No context built for run_id={run_id}, using original query",
        )
        return query
