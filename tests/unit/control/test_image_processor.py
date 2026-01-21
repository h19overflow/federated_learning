"""
Unit tests for ImageProcessor component.
Tests image I/O operations including reading from uploads and base64 conversion.
"""

import base64

import pytest
from fastapi import HTTPException, UploadFile

from federated_pneumonia_detection.src.control.model_inferance.internals.image_processor import (
    ImageProcessor,
)


class TestImageProcessor:
    """Tests for ImageProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create ImageProcessor instance."""
        return ImageProcessor()

    # =========================================================================
    # Test read_from_upload method
    # =========================================================================

    @pytest.mark.asyncio
    async def test_read_from_upload_jpeg(self, processor, mock_upload_file_jpeg):
        """Test reading JPEG image from upload."""
        image = await processor.read_from_upload(mock_upload_file_jpeg)

        from PIL import Image

        assert isinstance(image, Image.Image)
        assert image.size[0] > 0
        assert image.size[1] > 0

    @pytest.mark.asyncio
    async def test_read_from_upload_png(self, processor, mock_upload_file_png):
        """Test reading PNG image from upload."""
        image = await processor.read_from_upload(mock_upload_file_png)

        from PIL import Image

        assert isinstance(image, Image.Image)
        assert image.size[0] > 0
        assert image.size[1] > 0

    @pytest.mark.asyncio
    async def test_read_from_upload_with_convert_rgb(
        self,
        processor,
        mock_upload_file_jpeg,
    ):
        """Test reading image and converting to RGB."""
        # Create a grayscale image
        from io import BytesIO

        import numpy as np
        from PIL import Image

        img_array = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        img = Image.fromarray(img_array, mode="L")

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        file = UploadFile(
            filename="test_gray.png",
            file=buffer,
            content_type="image/png",
        )

        # Read without conversion
        image_l = await processor.read_from_upload(file, convert_rgb=False)
        assert image_l.mode == "L"

        # Read with conversion
        buffer.seek(0)
        file = UploadFile(
            filename="test_gray.png",
            file=buffer,
            content_type="image/png",
        )
        image_rgb = await processor.read_from_upload(file, convert_rgb=True)
        assert image_rgb.mode == "RGB"

    @pytest.mark.asyncio
    async def test_read_from_upload_corrupted_image(
        self,
        processor,
        mock_upload_file_corrupted,
    ):
        """Test reading corrupted image raises HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            await processor.read_from_upload(mock_upload_file_corrupted)

        assert exc_info.value.status_code == 400
        assert "Failed to process image" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_read_from_upload_invalid_image_data(self, processor):
        """Test reading invalid image data raises HTTPException."""
        from io import BytesIO

        file = UploadFile(
            filename="invalid.jpg",
            file=BytesIO(b"not a real image"),
            content_type="image/jpeg",
        )

        with pytest.raises(HTTPException) as exc_info:
            await processor.read_from_upload(file)

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_read_from_upload_empty_file(self, processor):
        """Test reading empty file raises HTTPException."""
        from io import BytesIO

        file = UploadFile(
            filename="empty.jpg",
            file=BytesIO(b""),
            content_type="image/jpeg",
        )

        with pytest.raises(HTTPException):
            await processor.read_from_upload(file)

    @pytest.mark.asyncio
    async def test_read_from_upload_preserves_size(self, processor):
        """Test reading preserves original image dimensions."""
        from io import BytesIO

        import numpy as np
        from PIL import Image

        # Create image with specific size
        width, height = 300, 400
        img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, mode="RGB")

        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)

        file = UploadFile(
            filename="sized.jpg",
            file=buffer,
            content_type="image/jpeg",
        )

        image = await processor.read_from_upload(file)
        assert image.size == (width, height)

    # =========================================================================
    # Test to_base64 method
    # =========================================================================

    def test_to_base64_returns_string(self, processor, sample_xray_image):
        """Test to_base64 returns a string."""
        base64_str = processor.to_base64(sample_xray_image)
        assert isinstance(base64_str, str)
        assert len(base64_str) > 0

    def test_to_base64_valid_base64(self, processor, sample_xray_image):
        """Test to_base64 produces valid base64."""
        base64_str = processor.to_base64(sample_xray_image)

        # Should be decodable
        decoded = base64.b64decode(base64_str)
        assert len(decoded) > 0

    def test_to_base64_different_images_produce_different_strings(self, processor):
        """Test that different images produce different base64 strings."""
        import numpy as np
        from PIL import Image

        img1 = Image.fromarray(np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8))
        img2 = Image.fromarray(np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8))

        b64_1 = processor.to_base64(img1)
        b64_2 = processor.to_base64(img2)

        assert b64_1 != b64_2

    def test_to_base64_same_image_produces_same_string(self, processor):
        """Test that same image produces same base64 string."""
        import numpy as np
        from PIL import Image

        img_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)

        b64_1 = processor.to_base64(img)
        b64_2 = processor.to_base64(img)

        assert b64_1 == b64_2

    def test_to_base64_handles_grayscale(self, processor):
        """Test to_base64 handles grayscale images."""
        from PIL import Image

        img = Image.new("L", (100, 100), color=128)
        base64_str = processor.to_base64(img)

        assert isinstance(base64_str, str)
        decoded = base64.b64decode(base64_str)
        assert len(decoded) > 0

    def test_to_base64_handles_rgba(self, processor):
        """Test to_base64 handles RGBA images."""
        from PIL import Image

        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        base64_str = processor.to_base64(img)

        assert isinstance(base64_str, str)
        decoded = base64.b64decode(base64_str)
        assert len(decoded) > 0

    def test_to_base64_handles_rgb(self, processor):
        """Test to_base64 handles RGB images."""
        from PIL import Image

        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        base64_str = processor.to_base64(img)

        assert isinstance(base64_str, str)
        decoded = base64.b64decode(base64_str)
        assert len(decoded) > 0

    # =========================================================================
    # Test integration scenarios
    # =========================================================================

    @pytest.mark.asyncio
    async def test_round_trip_upload_to_base64(self, processor, mock_upload_file_jpeg):
        """Test complete round trip: upload -> read -> base64 -> decode."""
        # Read from upload
        image = await processor.read_from_upload(mock_upload_file_jpeg)

        # Convert to base64
        base64_str = processor.to_base64(image)

        # Decode back to image
        from io import BytesIO

        from PIL import Image

        decoded = base64.b64decode(base64_str)
        reconstructed = Image.open(BytesIO(decoded))

        # Images should have same dimensions
        assert reconstructed.size == image.size

    @pytest.mark.asyncio
    async def test_read_multiple_images_from_upload(self, processor):
        """Test reading multiple images sequentially."""
        from io import BytesIO

        import numpy as np
        from PIL import Image

        images = []
        for i in range(5):
            img_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)

            buffer = BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)

            file = UploadFile(
                filename=f"image_{i}.png",
                file=buffer,
                content_type="image/png",
            )

            image = await processor.read_from_upload(file)
            images.append(image)

        assert len(images) == 5
        for img in images:
            assert img.size == (100, 100)

    # =========================================================================
    # Test error handling edge cases
    # =========================================================================

    def test_to_base64_with_none_image(self, processor):
        """Test to_base64 with None raises AttributeError."""
        with pytest.raises(AttributeError):
            processor.to_base64(None)

    @pytest.mark.asyncio
    async def test_read_from_upload_with_file_seek_issues(self, processor):
        """Test reading when file position is at end."""
        from io import BytesIO

        import numpy as np
        from PIL import Image

        img_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        # Don't seek back to start

        file = UploadFile(
            filename="test.png",
            file=buffer,
            content_type="image/png",
        )

        # Should still work because read() is called which advances position
        image = await processor.read_from_upload(file)
        assert image is not None
