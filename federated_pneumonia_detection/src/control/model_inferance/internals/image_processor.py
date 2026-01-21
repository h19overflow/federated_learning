"""Image I/O processing component."""

import base64
from io import BytesIO

from fastapi import HTTPException, UploadFile
from PIL import Image


class ImageProcessor:
    """Handles image I/O operations."""

    async def read_from_upload(
        self,
        file: UploadFile,
        convert_rgb: bool = False,
    ) -> Image.Image:
        """Read image from uploaded file."""
        try:
            contents = await file.read()
            image = Image.open(BytesIO(contents))
            if convert_rgb:
                image = image.convert("RGB")
            return image
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process image: {str(e)}",
            )

    def to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
