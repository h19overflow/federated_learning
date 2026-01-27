"""Global exception handler middleware for structured error responses.

This module provides centralized exception handling for the FastAPI application.
All exceptions are caught and transformed into consistent JSON responses.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from starlette.responses import JSONResponse

from federated_pneumonia_detection.src.api.endpoints.schema.error_schemas import (
    APIErrorResponse,
    ErrorCode,
    ErrorDetail,
    ValidationErrorDetail,
)

logger = logging.getLogger(__name__)


class APIException(Exception):  # noqa: N818
    """Base exception class for API errors.

    All application-specific exceptions should inherit from this class.
    It provides a structured way to create errors with proper HTTP status codes.

    Usage Example:
        raise APIException(
            code=ErrorCode.NOT_FOUND_ERROR,
            message="Resource not found",
            status_code=404,
            details={"resource_id": "123"}
        )
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize API exception.

        Args:
            code: Error code from ErrorCode enum
            message: Human-readable error message
            status_code: HTTP status code (default: 500)
            details: Additional error context
        """
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


def _build_error_response(
    request: Request,
    code: ErrorCode,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> JSONResponse:
    """Build a structured error response.

    Args:
        request: FastAPI request object
        code: Error code from ErrorCode enum
        message: Human-readable error message
        details: Additional error context

    Returns:
        JSONResponse with structured error format
    """
    error_detail = ErrorDetail(
        code=code,
        message=message,
        details=details,
        path=str(request.url.path),
    )

    response_data = APIErrorResponse(error=error_detail)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response_data.model_dump(mode="json"),
    )


async def api_exception_handler(request: Request, exc: APIException) -> JSONResponse:
    """Handle APIException instances.

    Transforms APIException into a structured JSON response with the
    appropriate HTTP status code.

    Args:
        request: FastAPI request object
        exc: APIException to handle

    Returns:
        JSONResponse with structured error format
    """
    logger.error(
        f"APIException: {exc.code.value} - {exc.message} | Path: {request.url.path} | Details: {exc.details}"  # noqa: E501
    )

    error_detail = ErrorDetail(
        code=exc.code,
        message=exc.message,
        details=exc.details,
        path=str(request.url.path),
    )

    response_data = APIErrorResponse(error=error_detail)

    return JSONResponse(
        status_code=exc.status_code,
        content=response_data.model_dump(mode="json"),
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic RequestValidationError instances.

    Transforms validation errors into structured responses with field-level details.

    Args:
        request: FastAPI request object
        exc: RequestValidationError to handle

    Returns:
        JSONResponse with 422 status and field-level error details
    """
    logger.warning(
        f"Validation error on {request.url.path}: {exc.errors()}",
        extra={"validation_errors": exc.errors()},
    )

    # Extract field-level validation errors
    field_errors = []
    for error in exc.errors():
        location = " -> ".join(str(loc) for loc in error["loc"] if loc != "body")
        field_errors.append(
            ValidationErrorDetail(
                field=error["loc"][-1] if error["loc"] else "unknown",
                message=error["msg"],
                location=location,
            )
        )

    error_detail = ErrorDetail(
        code=ErrorCode.VALIDATION_ERROR,
        message="Request validation failed",
        details={"fields": [fe.model_dump(mode="json") for fe in field_errors]},
        path=str(request.url.path),
    )

    response_data = APIErrorResponse(error=error_detail)

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=response_data.model_dump(mode="json"),
    )


async def pydantic_validation_exception_handler(
    request: Request, exc: ValidationError
) -> JSONResponse:
    """Handle Pydantic ValidationError instances.

    Handles validation errors that occur in Pydantic models outside of request body parsing.  # noqa: E501

    Args:
        request: FastAPI request object
        exc: ValidationError to handle

    Returns:
        JSONResponse with 422 status and field-level error details
    """
    logger.warning(
        f"Pydantic validation error on {request.url.path}: {exc.errors()}",
        extra={"validation_errors": exc.errors()},
    )

    # Extract field-level validation errors
    field_errors = []
    for error in exc.errors():
        location = " -> ".join(str(loc) for loc in error["loc"])
        field_errors.append(
            ValidationErrorDetail(
                field=error["loc"][-1] if error["loc"] else "unknown",
                message=error["msg"],
                location=location,
            )
        )

    error_detail = ErrorDetail(
        code=ErrorCode.VALIDATION_ERROR,
        message="Data validation failed",
        details={"fields": [fe.model_dump(mode="json") for fe in field_errors]},
        path=str(request.url.path),
    )

    response_data = APIErrorResponse(error=error_detail)

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=response_data.model_dump(mode="json"),
    )


async def database_exception_handler(
    request: Request, exc: SQLAlchemyError
) -> JSONResponse:
    """Handle SQLAlchemy database errors.

    Catches database exceptions and returns a generic error without
    exposing sensitive database details to the client.

    Args:
        request: FastAPI request object
        exc: SQLAlchemyError to handle

    Returns:
        JSONResponse with 500 status
    """
    logger.error(
        f"Database error on {request.url.path}: {str(exc)}",
        exc_info=True,
    )

    error_detail = ErrorDetail(
        code=ErrorCode.DATABASE_ERROR,
        message="A database error occurred",
        path=str(request.url.path),
    )

    response_data = APIErrorResponse(error=error_detail)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response_data.model_dump(mode="json"),
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all other uncaught exceptions.

    This is a catch-all handler that logs unexpected errors and returns
    a generic error response without exposing implementation details.

    Args:
        request: FastAPI request object
        exc: Exception to handle

    Returns:
        JSONResponse with 500 status
    """
    logger.error(
        f"Unhandled exception on {request.url.path}: {type(exc).__name__}: {str(exc)}",
        exc_info=True,
    )

    error_detail = ErrorDetail(
        code=ErrorCode.SYSTEM_ERROR,
        message="An unexpected error occurred",
        path=str(request.url.path),
    )

    response_data = APIErrorResponse(error=error_detail)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response_data.model_dump(mode="json"),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers with the FastAPI application.

    This function should be called after app creation to enable
    structured error handling across the entire application.

    Usage:
        app = FastAPI()
        register_exception_handlers(app)

    Args:
        app: FastAPI application instance
    """
    # Custom APIException
    app.add_exception_handler(APIException, api_exception_handler)

    # Pydantic validation errors
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ValidationError, pydantic_validation_exception_handler)

    # SQLAlchemy database errors
    app.add_exception_handler(SQLAlchemyError, database_exception_handler)

    # Catch-all for any other exceptions
    app.add_exception_handler(Exception, generic_exception_handler)

    logger.info("Registered global exception handlers")
