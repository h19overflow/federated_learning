"""Analytics control layer - Facade pattern for business logic.

This module provides a clean facade interface for analytics operations
with dependency injection and caching. All implementation details are
hidden in the internals submodule.

Usage:
    from federated_pneumonia_detection.src.control.analytics import AnalyticsFacade

    # In startup.py:
    analytics = AnalyticsFacade(...)
    app.state.analytics = analytics

    # In endpoints:
    analytics = Depends(get_analytics)
    analytics.metrics.get_analytics_summary(...)
"""

from .facade import AnalyticsFacade

__all__ = ["AnalyticsFacade"]
