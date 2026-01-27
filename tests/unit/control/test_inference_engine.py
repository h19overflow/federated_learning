"""
Unit tests for InferenceEngine component.
Tests model loading, preprocessing, and prediction logic with mocking.
"""

from unittest.mock import Mock, patch

import pytest
import torch

from federated_pneumonia_detection.src.control.model_inferance.internals.inference_engine import (  # noqa: E501
    InferenceEngine,
)


class TestInferenceEngine:
    """Tests for InferenceEngine class."""

    # =========================================================================
    # Test initialization
    # =========================================================================

    @pytest.fixture
    def mock_lit_model_class(self):
        """Mock LitResNetEnhanced.load_from_checkpoint."""
        with patch(
            "federated_pneumonia_detection.src.control.dl_model.internals.model."
            "lit_resnet_enhanced.LitResNetEnhanced",
        ) as mock_lit:
            mock_model = Mock()
            mock_model.eval = Mock()
            mock_model.freeze = Mock()
            mock_model.to = Mock(return_value=mock_model)
            mock_lit.load_from_checkpoint = Mock(return_value=mock_model)
            yield mock_lit, mock_model

    def test_init_with_default_checkpoint(self, mock_lit_model_class, tmp_path):
        """Test initialization with default checkpoint path."""
        checkpoint_dir = tmp_path / "model_inferance"
        checkpoint_dir.mkdir(parents=True)
        checkpoint_file = checkpoint_dir / "pneumonia_model_07_0.928.ckpt"
        checkpoint_file.write_text("mock checkpoint")

        with patch.object(InferenceEngine, "DEFAULT_CHECKPOINT_PATH", checkpoint_file):
            engine = InferenceEngine()

            assert engine.checkpoint_path == checkpoint_file
            assert engine.device in ["cpu", "cuda"]
            assert engine.model is not None
            assert engine.model_version == checkpoint_file.stem

    def test_init_with_custom_checkpoint(self, mock_lit_model_class, tmp_path):
        """Test initialization with custom checkpoint path."""
        mock_lit, mock_model = mock_lit_model_class

        checkpoint_file = tmp_path / "custom_model.ckpt"
        checkpoint_file.write_text("mock")

        with patch("torch.cuda.is_available", return_value=False):
            engine = InferenceEngine(checkpoint_path=checkpoint_file, device="cpu")

            assert engine.checkpoint_path == checkpoint_file
            assert engine.device == "cpu"
            mock_lit.load_from_checkpoint.assert_called_once()

    def test_init_with_gpu_device(self, mock_lit_model_class, tmp_path):
        """Test initialization with GPU device."""
        checkpoint_file = tmp_path / "custom_model.ckpt"
        checkpoint_file.write_text("mock")

        engine = InferenceEngine(checkpoint_path=checkpoint_file, device="cuda")

        assert engine.device == "cuda"

    def test_init_model_not_found_raises_error(self, tmp_path):
        """Test initialization with missing checkpoint raises FileNotFoundError."""
        nonexistent = tmp_path / "nonexistent.ckpt"

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            InferenceEngine(checkpoint_path=nonexistent)

    # =========================================================================
    # Test model loading
    # =========================================================================

    def test_load_model_calls_load_from_checkpoint(
        self,
        mock_lit_model_class,
        tmp_path,
    ):
        """Test that _load_model correctly calls Lightning's load_from_checkpoint."""
        mock_lit, _ = mock_lit_model_class

        checkpoint_file = tmp_path / "test.ckpt"
        checkpoint_file.write_text("mock")

        InferenceEngine(checkpoint_path=checkpoint_file, device="cpu")

        mock_lit.load_from_checkpoint.assert_called_once()
        call_args = mock_lit.load_from_checkpoint.call_args
        assert str(checkpoint_file) in call_args[0]
        assert "map_location" in call_args[1]
        assert call_args[1]["map_location"] == "cpu"

    def test_load_model_sets_eval_mode(self, mock_lit_model_class, tmp_path):
        """Test that loaded model is set to eval mode."""
        _, mock_model = mock_lit_model_class

        checkpoint_file = tmp_path / "test.ckpt"
        checkpoint_file.write_text("mock")

        InferenceEngine(checkpoint_path=checkpoint_file, device="cpu")

        mock_model.eval.assert_called_once()

    def test_load_model_freezes_model(self, mock_lit_model_class, tmp_path):
        """Test that loaded model is frozen."""
        _, mock_model = mock_lit_model_class

        checkpoint_file = tmp_path / "test.ckpt"
        checkpoint_file.write_text("mock")

        InferenceEngine(checkpoint_path=checkpoint_file, device="cpu")

        mock_model.freeze.assert_called_once()

    def test_load_model_moves_to_device(self, mock_lit_model_class, tmp_path):
        """Test that model is moved to specified device."""
        _, mock_model = mock_lit_model_class

        checkpoint_file = tmp_path / "test.ckpt"
        checkpoint_file.write_text("mock")

        InferenceEngine(checkpoint_path=checkpoint_file, device="cpu")

        mock_model.to.assert_called_once()
        assert mock_model.to.call_args[0][0] == "cpu"

    # =========================================================================
    # Test transform setup
    # =========================================================================

    def test_setup_transforms_creates_transform(self, mock_lit_model_class, tmp_path):
        """Test that _setup_transforms creates transform pipeline."""
        checkpoint_file = tmp_path / "test.ckpt"
        checkpoint_file.write_text("mock")

        engine = InferenceEngine(checkpoint_path=checkpoint_file, device="cpu")

        assert engine._transform is not None

    def test_transform_applies_to_image(
        self,
        mock_lit_model_class,
        tmp_path,
        sample_rgb_image,
    ):
        """Test that transform can be applied to an image."""
        checkpoint_file = tmp_path / "test.ckpt"
        checkpoint_file.write_text("mock")

        engine = InferenceEngine(checkpoint_path=checkpoint_file, device="cpu")

        tensor = engine._transform(sample_rgb_image)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[0] == 3  # RGB channels
        assert tensor.shape[1] == 224  # Height
        assert tensor.shape[2] == 224  # Width

    # =========================================================================
    # Test preprocessing
    # =========================================================================

    def test_preprocess_rgb_image(
        self,
        mock_lit_model_class,
        tmp_path,
        sample_rgb_image,
    ):
        """Test preprocessing RGB image."""
        checkpoint_file = tmp_path / "test.ckpt"
        checkpoint_file.write_text("mock")

        engine = InferenceEngine(checkpoint_path=checkpoint_file, device="cpu")

        tensor = engine.preprocess(sample_rgb_image)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)  # Batch, channels, height, width
        assert tensor.device.type == "cpu"

    def test_preprocess_grayscale_converts_to_rgb(
        self,
        mock_lit_model_class,
        tmp_path,
        sample_xray_image,
    ):
        """Test preprocessing converts grayscale to RGB."""
        checkpoint_file = tmp_path / "test.ckpt"
        checkpoint_file.write_text("mock")

        engine = InferenceEngine(checkpoint_path=checkpoint_file, device="cpu")

        tensor = engine.preprocess(sample_xray_image)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)  # Should be RGB

    def test_preprocess_resize_image(self, mock_lit_model_class, tmp_path):
        """Test preprocessing resizes image to 224x224."""
        import numpy as np
        from PIL import Image

        checkpoint_file = tmp_path / "test.ckpt"
        checkpoint_file.write_text("mock")

        engine = InferenceEngine(checkpoint_path=checkpoint_file, device="cpu")

        # Create large image
        large_img = Image.fromarray(
            np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8),
        )

        tensor = engine.preprocess(large_img)

        assert tensor.shape[2] == 224  # Height
        assert tensor.shape[3] == 224  # Width

    def test_preprocess_adds_batch_dimension(
        self,
        mock_lit_model_class,
        tmp_path,
        sample_rgb_image,
    ):
        """Test preprocessing adds batch dimension."""
        checkpoint_file = tmp_path / "test.ckpt"
        checkpoint_file.write_text("mock")

        engine = InferenceEngine(checkpoint_path=checkpoint_file, device="cpu")

        tensor = engine.preprocess(sample_rgb_image)

        assert tensor.shape[0] == 1  # Batch dimension

    # =========================================================================
    # Test prediction
    # =========================================================================

    def test_predict_returns_tuple(
        self,
        mock_lit_model_class,
        tmp_path,
        sample_rgb_image,
    ):
        """Test predict returns tuple of 4 elements."""
        mock_lit, mock_model = mock_lit_model_class
        mock_model.return_value = torch.tensor([[0.7]])

        checkpoint_file = tmp_path / "test.ckpt"
        checkpoint_file.write_text("mock")

        engine = InferenceEngine(checkpoint_path=checkpoint_file, device="cpu")

        result = engine.predict(sample_rgb_image)

        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_predict_pneumonia_high_confidence(
        self,
        mock_lit_model_class,
        tmp_path,
        sample_rgb_image,
    ):
        """Test prediction returns PNEUMONIA with high confidence."""
        mock_lit, mock_model = mock_lit_model_class
        # Logit for 0.85 probability: ln(0.85/0.15) ≈ 1.7346
        mock_model.return_value = torch.tensor([[1.7346]])

        checkpoint_file = tmp_path / "test.ckpt"
        checkpoint_file.write_text("mock")

        engine = InferenceEngine(checkpoint_path=checkpoint_file, device="cpu")

        pred_class, confidence, pneu_prob, norm_prob = engine.predict(sample_rgb_image)

        assert pred_class == "PNEUMONIA"
        assert confidence == pytest.approx(0.85, abs=0.01)
        assert pneu_prob == pytest.approx(0.85, abs=0.01)
        assert norm_prob == pytest.approx(0.15, abs=0.01)

    def test_predict_normal_high_confidence(
        self,
        mock_lit_model_class,
        tmp_path,
        sample_rgb_image,
    ):
        """Test prediction returns NORMAL with high confidence."""
        mock_lit, mock_model = mock_lit_model_class
        # Logit for 0.15 probability (Normal 0.85): ln(0.15/0.85) ≈ -1.7346
        mock_model.return_value = torch.tensor([[-1.7346]])  # Negative logit

        checkpoint_file = tmp_path / "test.ckpt"
        checkpoint_file.write_text("mock")

        engine = InferenceEngine(checkpoint_path=checkpoint_file, device="cpu")

        pred_class, confidence, pneu_prob, norm_prob = engine.predict(sample_rgb_image)

        assert pred_class == "NORMAL"
        assert norm_prob == pytest.approx(0.85, abs=0.01)

    def test_predict_boundary_case(
        self,
        mock_lit_model_class,
        tmp_path,
        sample_rgb_image,
    ):
        """Test prediction at 0.5 boundary."""
        mock_lit, mock_model = mock_lit_model_class
        mock_model.return_value = torch.tensor([[0.0]])

        checkpoint_file = tmp_path / "test.ckpt"
        checkpoint_file.write_text("mock")

        engine = InferenceEngine(checkpoint_path=checkpoint_file, device="cpu")

        pred_class, confidence, pneu_prob, norm_prob = engine.predict(sample_rgb_image)

        assert pred_class == "PNEUMONIA"  # >= 0.5
        assert confidence == pytest.approx(0.5, abs=0.01)

    def test_predict_sigmoid_applied(
        self,
        mock_lit_model_class,
        tmp_path,
        sample_rgb_image,
    ):
        """Test that sigmoid is applied to logits."""
        mock_lit, mock_model = mock_lit_model_class
        mock_model.return_value = torch.tensor([[2.0]])

        checkpoint_file = tmp_path / "test.ckpt"
        checkpoint_file.write_text("mock")

        engine = InferenceEngine(checkpoint_path=checkpoint_file, device="cpu")

        _, _, pneu_prob, norm_prob = engine.predict(sample_rgb_image)

        # Sigmoid of 2.0 is approximately 0.88
        assert pneu_prob == pytest.approx(0.88, abs=0.01)
        assert norm_prob == pytest.approx(0.12, abs=0.01)

    # =========================================================================
    # Test properties and methods
    # =========================================================================

    def test_is_gpu_property(self, mock_lit_model_class, tmp_path):
        """Test is_gpu property."""
        checkpoint_file = tmp_path / "test.ckpt"
        checkpoint_file.write_text("mock")

        engine_cpu = InferenceEngine(checkpoint_path=checkpoint_file, device="cpu")
        assert engine_cpu.is_gpu is False

        engine_gpu = InferenceEngine(checkpoint_path=checkpoint_file, device="cuda")
        assert engine_gpu.is_gpu is True

    def test_get_info(self, mock_lit_model_class, tmp_path):
        """Test get_info returns correct information."""
        checkpoint_file = tmp_path / "test.ckpt"
        checkpoint_file.write_text("mock")

        with patch("torch.cuda.is_available", return_value=False):
            engine = InferenceEngine(checkpoint_path=checkpoint_file, device="cpu")

        info = engine.get_info()

        assert "model_version" in info
        assert "device" in info
        assert "gpu_available" in info
        assert "checkpoint_path" in info
        assert info["device"] == "cpu"
        assert info["gpu_available"] is False
        assert str(checkpoint_file) in info["checkpoint_path"]

    # =========================================================================
    # Test edge cases and error handling
    # =========================================================================

    def test_predict_with_none_model(self, tmp_path, sample_rgb_image):
        """Test predict with uninitialized model raises RuntimeError."""
        # Create engine without loading model
        engine = InferenceEngine.__new__(InferenceEngine)
        engine.model = None
        engine._transform = Mock()
        engine.device = "cpu"

        with pytest.raises(RuntimeError, match="Inference engine not available"):
            engine.predict(sample_rgb_image)

    def test_predict_with_tiny_image(self, mock_lit_model_class, tmp_path):
        """Test predict with very small image."""
        mock_lit, mock_model = mock_lit_model_class
        mock_model.return_value = torch.tensor([[0.6]])

        checkpoint_file = tmp_path / "test.ckpt"
        checkpoint_file.write_text("mock")

        engine = InferenceEngine(checkpoint_path=checkpoint_file, device="cpu")

        # Should still work, transform will resize
        from PIL import Image

        tiny_img = Image.new("RGB", (8, 8), color=(128, 128, 128))

        result = engine.predict(tiny_img)
        assert len(result) == 4

    def test_predict_with_extreme_logits(
        self,
        mock_lit_model_class,
        tmp_path,
        sample_rgb_image,
    ):
        """Test predict with extreme logit values."""
        mock_lit, mock_model = mock_lit_model_class

        # Test very positive
        mock_model.return_value = torch.tensor([[10.0]])
        checkpoint_file = tmp_path / "test.ckpt"
        checkpoint_file.write_text("mock")

        engine = InferenceEngine(checkpoint_path=checkpoint_file, device="cpu")

        _, _, pneu_prob, norm_prob = engine.predict(sample_rgb_image)

        # Sigmoid of 10 is very close to 1
        assert pneu_prob > 0.999
        assert norm_prob < 0.001

        # Test very negative
        mock_model.return_value = torch.tensor([[-10.0]])
        _, _, pneu_prob, norm_prob = engine.predict(sample_rgb_image)

        # Sigmoid of -10 is very close to 0
        assert pneu_prob < 0.001
        assert norm_prob > 0.999
