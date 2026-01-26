import pytest


def test_predict_valid_image(api_client_with_db, mock_inference_engine, dummy_image_bytes):
    """
    Positive test: Upload a dummy image and check for successful prediction.
    Ensures the endpoint correctly interacts with the InferenceService and engine.
    """
    # Arrange
    mock_inference_engine.predict.return_value = ("PNEUMONIA", 0.95, 0.95, 0.05)

    # Act
    # Simulate file upload with valid image bytes
    response = api_client_with_db.post(
        "/api/inference/predict",
        files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
    )

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["prediction"]["predicted_class"] == "PNEUMONIA"
    assert data["prediction"]["confidence"] == 0.95
    assert data["model_version"] == "mock-1.0.0"

    # Verify mock was called
    assert mock_inference_engine.predict.called


def test_predict_invalid_file(api_client_with_db):
    """
    Negative test: Upload a non-image file and check for 400 Bad Request.
    Ensures the ImageValidator correctly rejects unsupported file types.
    """
    # Arrange
    dummy_text_content = b"this is not an image"

    # Act
    response = api_client_with_db.post(
        "/api/inference/predict",
        files={"file": ("test.txt", dummy_text_content, "text/plain")},
    )

    # Assert
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]
