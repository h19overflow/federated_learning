"""Ranking service for retrieving and caching top-performing runs."""

import json
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from federated_pneumonia_detection.src.internals.loggers.logger import get_logger

from ..infrastructure import CacheProvider

logger = get_logger(__name__)


class RankingService:
    """Provides ranking and caching of top runs by specified metrics."""

    def __init__(self, *, cache: CacheProvider, run_crud_obj: Any) -> None:
        """
        Initialize RankingService.

        Args:
            cache: CacheProvider instance for caching results
            run_crud_obj: CRUD object for run operations
        """
        self._cache = cache
        self._run_crud = run_crud_obj

    def top_runs(
        self,
        db: Session,
        *,
        metric: str,
        n: int,
        filters: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Get top N runs sorted by metric, with caching.

        Args:
            db: Database session
            metric: Metric key to sort by (e.g., 'best_accuracy')
            n: Number of top runs to return
            filters: Optional filter dictionary for run queries

        Returns:
            List of top N run dictionaries sorted by metric descending
        """
        filters = filters or {}

        # Generate cache key from metric, n, and filters
        cache_key = self._generate_cache_key(metric=metric, n=n, filters=filters)

        # Try to retrieve from cache
        cached_result = self._cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"[RankingService] Cache hit for key: {cache_key}")
            try:
                # Validate json.loads result is a list before returning
                parsed_result = json.loads(cached_result)
                if isinstance(parsed_result, list):
                    return parsed_result
                else:
                    logger.warning(
                        f"[RankingService] Invalid cached format for key: {cache_key}"
                    )
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"[RankingService] Failed to parse cached result: {e}")

        logger.info(f"[RankingService] Cache miss for key: {cache_key}")

        # Fetch runs from database
        try:
            runs = self._run_crud.read_all(db=db, filters=filters)
            if not runs:
                logger.warning(
                    f"[RankingService] No runs found with filters: {filters}"
                )
                return []

            # Sort by metric and limit to n
            top_runs = sorted(
                runs,
                key=lambda x: x.get(metric, 0.0),
                reverse=True,
            )[:n]

            # Cache the result
            self._cache.set(cache_key, json.dumps(top_runs))
            logger.info(
                f"[RankingService] Cached {len(top_runs)} top runs for key: {cache_key}"
            )

            return top_runs

        except Exception as err:
            logger.error(f"[RankingService] Error fetching top runs: {err}")
            raise

    def _generate_cache_key(
        self, *, metric: str, n: int, filters: Dict[str, Any]
    ) -> str:
        """
        Generate a cache key from metric, n, and filters.

        Args:
            metric: Metric key
            n: Number of runs
            filters: Filter dictionary

        Returns:
            Cache key string
        """
        filters_str = json.dumps(filters, sort_keys=True)
        return f"top_runs:{metric}:{n}:{filters_str}"

    def invalidate_cache(
        self, *, metric: str | None = None, n: int | None = None
    ) -> None:
        """
        Invalidate cache entries.

        Args:
            metric: Optional metric to invalidate (if None, invalidates all)
            n: Optional n value to invalidate (if None, invalidates all)
        """
        if metric is None or n is None:
            logger.info("[RankingService] Invalidating all ranking cache")
            self._cache.clear()
        else:
            # Could implement pattern-based invalidation here
            logger.info(
                f"[RankingService] Invalidating cache for metric={metric}, n={n}"
            )
