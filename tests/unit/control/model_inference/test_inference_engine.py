from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image
from torchvision import transforms

# Import the class under test
from federated_pneumonia_detection.src.control.model_inferance.internals.inference_engine import (  # noqa: E501
    InferenceEngine,
)


@pytest.fixture
def mock_model():
    """Fixture for the mocked model."""
    model = MagicMock()
    model.to.return_value = model
    return model


@pytest.fixture
def mock_lit_resnet(mock_model):
    """Fixture for mocking LitResNetEnhanced."""
    # LitResNetEnhanced is imported inside _load_model from its original location
    with patch(
        "federated_pneumonia_detection.src.control.dl_model.internals.model."
        "lit_resnet_enhanced.LitResNetEnhanced"
    ) as mock:
        mock.load_from_checkpoint.return_value = mock_model
        yield mock


@pytest.fixture
def dummy_image():
    """Fixture for a dummy PIL image."""
    return Image.new("RGB", (100, 100), color="red")


class TestInferenceEngine:
    """Test suite for InferenceEngine."""

    @patch("pathlib.Path.exists")
    @patch("torch.cuda.is_available")
    def test_initialization(self, mock_cuda, mock_exists, mock_lit_resnet, mock_model):
        """Test initialization, model loading, and eval mode."""
        mock_exists.return_value = True
        mock_cuda.return_value = False
        checkpoint_path = Path("fake/path.ckpt")

        engine = InferenceEngine(checkpoint_path=checkpoint_path, device="cpu")

        # Verify load_from_checkpoint called with correct args
        mock_lit_resnet.load_from_checkpoint.assert_called_once_with(
            str(checkpoint_path),
            map_location="cpu",
            config=None,
            class_weights_tensor=None,
        )

        # Verify model set to eval mode and frozen
        mock_model.eval.assert_called_once()
        mock_model.freeze.assert_called_once()
        mock_model.to.assert_called_with("cpu")
        assert engine.device == "cpu"
        assert engine.model == mock_model

    @patch("pathlib.Path.exists")
    @patch("torch.cuda.is_available")
    def test_predict_happy_path(
        self, mock_cuda, mock_exists, mock_lit_resnet, mock_model, dummy_image
    ):
        """Test prediction with a positive result (PNEUMONIA)."""
        mock_exists.return_value = True
        mock_cuda.return_value = False

        # Mock model forward pass to return a logit that results in PNEUMONIA
        # The implementation uses sigmoid for binary classification.
        # sigmoid(2.0) ~= 0.8808
        expected_logit = 2.0
        mock_model.return_value = torch.tensor([expected_logit])

        engine = InferenceEngine(checkpoint_path=Path("fake.ckpt"), device="cpu")

        # Mock Image.open as requested
        with patch("PIL.Image.open", return_value=dummy_image) as mock_open:
            img = Image.open("dummy.jpg")
            label, confidence, prob_p, prob_n = engine.predict(img)

            mock_open.assert_called_once_with("dummy.jpg")

        expected_prob_p = torch.sigmoid(torch.tensor(expected_logit)).item()

        assert label == "PNEUMONIA"
        assert confidence == pytest.approx(expected_prob_p)
        assert prob_p == pytest.approx(expected_prob_p)
        assert prob_n == pytest.approx(1.0 - expected_prob_p)

    @patch("pathlib.Path.exists")
    @patch("torch.cuda.is_available")
    def test_predict_normal_class(
        self, mock_cuda, mock_exists, mock_lit_resnet, mock_model, dummy_image
    ):
        """Test prediction with a negative result (NORMAL)."""
        mock_exists.return_value = True
        mock_cuda.return_value = False

        # Mock model forward pass to return a logit that results in NORMAL
        # sigmoid(-2.0) ~= 0.1192
        expected_logit = -2.0
        mock_model.return_value = torch.tensor([expected_logit])

        engine = InferenceEngine(checkpoint_path=Path("fake.ckpt"), device="cpu")
        label, confidence, prob_p, prob_n = engine.predict(dummy_image)

        expected_prob_p = torch.sigmoid(torch.tensor(expected_logit)).item()
        expected_prob_n = 1.0 - expected_prob_p

        assert label == "NORMAL"
        assert prob_p == pytest.approx(expected_prob_p)
        assert prob_n == pytest.approx(expected_prob_n)
        assert confidence == pytest.approx(expected_prob_n)

    @patch("pathlib.Path.exists")
    @patch("torch.cuda.is_available")
    def test_preprocessing_transforms(
        self, mock_cuda, mock_exists, mock_lit_resnet, mock_model, dummy_image
    ):
        """Test that preprocessing transforms are applied correctly."""
        mock_exists.return_value = True
        mock_cuda.return_value = False
        engine = InferenceEngine(checkpoint_path=Path("fake.ckpt"), device="cpu")

        # Verify transforms are set up with correct types
        transform_types = [type(t) for t in engine._transform.transforms]
        assert transforms.Resize in transform_types
        assert transforms.CenterCrop in transform_types
        assert transforms.ToTensor in transform_types
        assert transforms.Normalize in transform_types

        # Test preprocess method
        input_tensor = engine.preprocess(dummy_image)

        # Verify shape (batch, channels, height, width) -> (1, 3, 224, 224)
        assert input_tensor.shape == (1, 3, 224, 224)
        assert isinstance(input_tensor, torch.Tensor)

        # Verify image conversion to RGB if needed
        gray_img = Image.new("L", (100, 100), color=128)
        with patch.object(
            Image.Image, "convert", wraps=gray_img.convert
        ) as mock_convert:
            engine.preprocess(gray_img)
            mock_convert.assert_called_with("RGB")

    @patch("pathlib.Path.exists")
    @patch("torch.cuda.is_available")
    def test_predict_exception_corrupted_image(
        self, mock_cuda, mock_exists, mock_lit_resnet, mock_model
    ):
        """Test handling of corrupted images via Image.open failure."""
        mock_exists.return_value = True
        mock_cuda.return_value = False
        engine = InferenceEngine(checkpoint_path=Path("fake.ckpt"), device="cpu")

        # Mock Image.open to raise an error as requested
        with patch("PIL.Image.open", side_effect=IOError("Corrupted image")):
            with pytest.raises(IOError, match="Corrupted image"):
                img = Image.open("corrupted.jpg")
                engine.predict(img)

    @patch("pathlib.Path.exists")
    @patch("torch.cuda.is_available")
    def test_get_info(self, mock_cuda, mock_exists, mock_lit_resnet, mock_model):
        """Test engine info retrieval."""
        mock_exists.return_value = True
        mock_cuda.return_value = False
        checkpoint_path = Path("my_model_v1.ckpt")
        engine = InferenceEngine(checkpoint_path=checkpoint_path, device="cpu")

        info = engine.get_info()
        assert info["model_version"] == "my_model_v1"
        assert info["device"] == "cpu"
        assert info["checkpoint_path"] == str(checkpoint_path)
