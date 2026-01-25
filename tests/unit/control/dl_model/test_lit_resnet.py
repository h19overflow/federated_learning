import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch, ANY
from federated_pneumonia_detection.src.control.dl_model.internals.model.lit_resnet_enhanced import (
    LitResNetEnhanced,
)
from federated_pneumonia_detection.src.control.dl_model.internals.model.losses import (
    FocalLoss,
    FocalLossWithLabelSmoothing,
)


@pytest.fixture
def mock_resnet_head():
    with patch(
        "federated_pneumonia_detection.src.control.dl_model.internals.model.lit_resnet_enhanced.ResNetWithCustomHead"
    ) as mock:
        instance = mock.return_value
        instance.get_model_info.return_value = {"total_parameters": 1000}
        # Add dummy parameters for optimizer initialization
        instance.parameters.return_value = [nn.Parameter(torch.randn(1, 1))]
        yield mock


class TestLitResNetEnhanced:
    def test_initialization_success(self, mock_config, mock_resnet_head):
        """Verify successful initialization with default parameters."""
        # Default label_smoothing is 0.1 in LitResNetEnhanced
        model = LitResNetEnhanced(config=mock_config)

        assert model.config == mock_config
        assert isinstance(model.loss_factory, FocalLossWithLabelSmoothing)
        mock_resnet_head.assert_called_once()

    def test_initialization_no_smoothing(self, mock_config, mock_resnet_head):
        """Verify FocalLoss is used when label_smoothing is 0."""
        model = LitResNetEnhanced(config=mock_config, label_smoothing=0)
        assert isinstance(model.loss_factory, FocalLoss)

    def test_initialization_invalid_lr(self, mock_config, mock_resnet_head):
        """Verify ValueError is raised for invalid learning rate."""
        mock_config.get.side_effect = (
            lambda key, default=None: 0 if key == "experiment.learning_rate" else 0.0001
        )

        with pytest.raises(ValueError, match="Learning rate must be positive"):
            LitResNetEnhanced(config=mock_config)

    def test_initialization_invalid_wd(self, mock_config, mock_resnet_head):
        """Verify ValueError is raised for invalid weight decay."""
        mock_config.get.side_effect = (
            lambda key, default=None: -1 if key == "experiment.weight_decay" else 0.001
        )

        with pytest.raises(ValueError, match="Weight decay must be non-negative"):
            LitResNetEnhanced(config=mock_config)

    def test_loss_selection_focal_smoothing(self, mock_config, mock_resnet_head):
        """Verify FocalLossWithLabelSmoothing is selected when label_smoothing > 0."""
        model = LitResNetEnhanced(
            config=mock_config, use_focal_loss=True, label_smoothing=0.1
        )
        assert isinstance(model.loss_factory, FocalLossWithLabelSmoothing)

    def test_loss_selection_bce(self, mock_config, mock_resnet_head):
        """Verify BCEWithLogitsLoss is selected when use_focal_loss=False."""
        model = LitResNetEnhanced(config=mock_config, use_focal_loss=False)
        assert isinstance(model.loss_factory, nn.BCEWithLogitsLoss)

    def test_forward(self, mock_config, mock_resnet_head):
        """Verify forward calls internal model."""
        model = LitResNetEnhanced(config=mock_config)
        x = torch.randn(1, 3, 224, 224)
        model(x)
        model.model.assert_called_once_with(x)

    def test_training_step(self, mock_config, mock_resnet_head):
        """Verify training_step returns loss and logs metrics."""
        model = LitResNetEnhanced(config=mock_config)
        model.log = MagicMock()

        # Mock model output
        model.model.return_value = torch.tensor([[0.5]])

        batch = (torch.randn(1, 3, 224, 224), torch.tensor([1]))
        loss = model.training_step(batch, 0)

        assert isinstance(loss, torch.Tensor)
        assert model.log.called
        # Check if train_loss was logged
        model.log.assert_any_call(
            "train_loss",
            ANY,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def test_validation_step(self, mock_config, mock_resnet_head):
        """Verify validation_step logs metrics."""
        model = LitResNetEnhanced(config=mock_config)
        model.log = MagicMock()
        model.model.return_value = torch.tensor([[0.5]])

        batch = (torch.randn(1, 3, 224, 224), torch.tensor([1]))
        loss = model.validation_step(batch, 0)

        assert isinstance(loss, torch.Tensor)
        model.log.assert_any_call(
            "val_loss", ANY, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )
        model.log.assert_any_call(
            "val_acc", ANY, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )

    def test_test_step(self, mock_config, mock_resnet_head):
        """Verify test_step logs metrics."""
        model = LitResNetEnhanced(config=mock_config)
        model.log = MagicMock()
        model.model.return_value = torch.tensor([[0.5]])

        batch = (torch.randn(1, 3, 224, 224), torch.tensor([1]))
        loss = model.test_step(batch, 0)

        assert isinstance(loss, torch.Tensor)
        model.log.assert_any_call(
            "test_loss", ANY, on_step=False, on_epoch=True, sync_dist=True
        )

    def test_configure_optimizers_cosine(self, mock_config, mock_resnet_head):
        """Verify return structure for cosine scheduler."""
        model = LitResNetEnhanced(config=mock_config, use_cosine_scheduler=True)
        with patch.object(
            model, "parameters", return_value=[nn.Parameter(torch.randn(1, 1))]
        ):
            opt_config = model.configure_optimizers()

        assert "optimizer" in opt_config
        assert "lr_scheduler" in opt_config
        assert isinstance(
            opt_config["lr_scheduler"]["scheduler"],
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        )

    def test_configure_optimizers_plateau(self, mock_config, mock_resnet_head):
        """Verify return structure for ReduceLROnPlateau."""
        model = LitResNetEnhanced(config=mock_config, use_cosine_scheduler=False)
        with patch.object(
            model, "parameters", return_value=[nn.Parameter(torch.randn(1, 1))]
        ):
            opt_config = model.configure_optimizers()

        assert "optimizer" in opt_config
        assert "lr_scheduler" in opt_config
        assert isinstance(
            opt_config["lr_scheduler"]["scheduler"],
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        )
        assert opt_config["lr_scheduler"]["monitor"] == "val_recall"

    def test_progressive_unfreeze(self, mock_config, mock_resnet_head):
        """Verify it calls _unfreeze_last_n_layers on internal model."""
        model = LitResNetEnhanced(config=mock_config)
        model.progressive_unfreeze(layers_to_unfreeze=2)

        assert model.unfrozen_layers == 2
        model.model._unfreeze_last_n_layers.assert_called_once_with(2)

    def test_freeze_backbone(self, mock_config, mock_resnet_head):
        """Verify freeze_backbone calls internal model."""
        model = LitResNetEnhanced(config=mock_config)
        model.unfrozen_layers = 5
        model.freeze_backbone()

        assert model.unfrozen_layers == 0
        model.model.freeze_backbone.assert_called_once()

    def test_unfreeze_backbone(self, mock_config, mock_resnet_head):
        """Verify unfreeze_backbone calls internal model."""
        model = LitResNetEnhanced(config=mock_config)
        model.unfreeze_backbone()
        model.model.unfreeze_backbone.assert_called_once()
