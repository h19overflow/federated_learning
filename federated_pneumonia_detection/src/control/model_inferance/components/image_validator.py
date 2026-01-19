"""Image file validation component."""

from typing import Optional

from fastapi import HTTPException, UploadFile


class ImageValidator:
    """Validates uploaded image files."""

    ALLOWED_CONTENT_TYPES = ["image/png", "image/jpeg", "image/jpg"]

    def validate(self, file: UploadFile) -> Optional[str]:
        """Return error message if invalid, None if valid."""
        if file.content_type not in self.ALLOWED_CONTENT_TYPES:
            return f"Invalid file type: {file.content_type}. Must be PNG or JPEG."
        return None

    def validate_or_raise(self, file: UploadFile) -> None:
        """Raise HTTPException if file type is invalid."""
        error = self.validate(file)
        if error:
            raise HTTPException(status_code=400, detail=error)
