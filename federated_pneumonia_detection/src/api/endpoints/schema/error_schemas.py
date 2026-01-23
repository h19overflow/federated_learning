"""Error schemas for structured error responses.

This module defines the error response structure used across the API.
All exceptions should return responses following this schema for consistency.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ErrorCode(str, Enum):
    """Error code categories for API errors.

    These codes provide machine-readable error types for client handling.
    """

    VALIDATION_ERROR = "VALIDATION_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    BUSINESS_LOGIC_ERROR = "BUSINESS_LOGIC_ERROR"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"
    NOT_FOUND_ERROR = "NOT_FOUND_ERROR"
    CONFLICT_ERROR = "CONFLICT_ERROR"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"


class ValidationErrorDetail(BaseModel):
    """Field-level validation error details.

    Used for Pydantic validation errors to provide specific field information.
    """

    field: str = Field(..., description="The field name that failed validation")
    message: str = Field(..., description="Validation error message for the field")
    location: Optional[str] = Field(
        None, description="Location of the field (e.g., 'body', 'query', 'path')"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "field": "email",
                "message": "Invalid email format",
                "location": "body",
            }
        }


class APIErrorResponse(BaseModel):
    """Standardized API error response.

    This structure is used for all error responses in the API.
    """

    error: "ErrorDetail" = Field(..., description="Error details")

    class Config:
        json_schema_extra = {
            "example": {
                "error": {
                    "code": "NOT_FOUND_ERROR",
                    "message": "Resource not found",
                    "details": {"resource_id": "123"},
                    "timestamp": "2024-01-23T12:34:56.789Z",
                    "path": "/api/resource/123",
                }
            }
        }


class ErrorDetail(BaseModel):
    """Detailed error information.

    Contains the error code, message, and optional context.
    """

    code: ErrorCode = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error context"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Error timestamp in UTC"
    )
    path: Optional[str] = Field(None, description="Request path that caused the error")

    class Config:
        json_schema_extra = {
            "example": {
                "code": "NOT_FOUND_ERROR",
                "message": "Resource not found",
                "details": {"resource_id": "123"},
                "timestamp": "2024-01-23T12:34:56.789Z",
                "path": "/api/resource/123",
            }
        }


# Update forward references for nested models
APIErrorResponse.model_rebuild()
