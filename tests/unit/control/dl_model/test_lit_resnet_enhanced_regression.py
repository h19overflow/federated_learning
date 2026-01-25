import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch, ANY
import torchmetrics
from federated_pneumonia_detection.src.control.dl_model.internals.model.lit_resnet_enhanced import (
    LitResNetEnhanced,
)
from federated_pneumonia_detection.src.control.dl_model.internals.model.losses import (
    FocalLoss,
    FocalLossWithLabelSmoothing,
)


@pytest.fixture
def mock_config():
    config = MagicMock()
    # Default values for ConfigManager.get
    config_values = {
        "experiment.dropout_rate": 0.5,
        "experiment.fine_tune_layers_count": 0,
        "experiment.use_torch_compile": False,
        "experiment.learning_rate": 0.001,
        "experiment.weight_decay": 0.0001,
        "experiment.epochs": 10,
        "experiment.min_lr": 1e-7,
        "experiment.reduce_lr_factor": 0.5,
        "experiment.reduce_lr_patience": 3,
    }
    config.get.side_effect = lambda key, default=None: config_values.get(key, default)
    return config


@pytest.fixture
def mock_resnet_head():
    with patch(
        "federated_pneumonia_detection.src.control.dl_model.internals.model.lit_resnet_enhanced.ResNetWithCustomHead"
    ) as mock:
        instance = mock.return_value
        instance.get_model_info.return_value = {"total_parameters": 1000}
        instance.parameters.return_value = [nn.Parameter(torch.randn(1, 1))]
        yield mock


class TestLitResNetEnhancedRegression:
    # --- 1. Metric Initialization and Updates ---

    def test_metrics_initialization(self, mock_config, mock_resnet_head):
        """Regression: Ensure all required metrics are initialized with correct tasks."""
        model = LitResNetEnhanced(config=mock_config)

        # Check training metrics
        assert model.metrics_handler.train_accuracy is not None
        assert model.metrics_handler.train_f1 is not None

        # Check validation metrics
        assert model.metrics_handler.val_accuracy is not None
        assert model.metrics_handler.val_precision is not None
        assert model.metrics_handler.val_recall is not None
        assert model.metrics_handler.val_f1 is not None
        assert model.metrics_handler.val_auroc is not None
        assert model.metrics_handler.val_confusion is not None

        # Check test metrics
        assert model.metrics_handler.test_accuracy is not None
        assert model.metrics_handler.test_f1 is not None

        # Verify it's a binary metric (or at least initialized correctly)
        assert "Binary" in model.metrics_handler.train_accuracy.__class__.__name__
        assert "Binary" in model.metrics_handler.val_auroc.__class__.__name__

    def test_metrics_updates_in_steps(self, mock_config, mock_resnet_head):
        """Regression: Ensure metrics are updated in each step with correct shapes."""
        model = LitResNetEnhanced(config=mock_config)
        model.log = MagicMock()

        # Mock metrics to track calls
        with (
            patch.object(
                model.metrics_handler.train_accuracy, "update"
            ) as mock_train_acc_update,
            patch.object(
                model.metrics_handler.val_accuracy, "update"
            ) as mock_val_acc_update,
            patch.object(
                model.metrics_handler.test_accuracy, "update"
            ) as mock_test_acc_update,
        ):
            batch = (torch.randn(2, 3, 224, 224), torch.tensor([0, 1]))

            # Training step
            model.model.return_value = torch.tensor([[0.1], [0.9]])
            model.training_step(batch, 0)
            # preds should be sigmoid of logits, targets should be int and unsqueezed
            mock_train_acc_update.assert_called_once()
            args, _ = mock_train_acc_update.call_args
            assert args[0].shape == (2, 1)  # preds
            assert args[1].shape == (2, 1)  # targets
            assert args[1].dtype == torch.int32

            # Validation step
            model.validation_step(batch, 0)
            mock_val_acc_update.assert_called_once()

            # Test step
            model.test_step(batch, 0)
            mock_test_acc_update.assert_called_once()

    def test_confusion_matrix_logging_and_reset(self, mock_config, mock_resnet_head):
        """Regression: Ensure confusion matrix is logged and reset at validation end."""
        model = LitResNetEnhanced(config=mock_config)
        model.log = MagicMock()

        # Mock confusion matrix compute and reset
        mock_cm = torch.tensor([[10, 2], [3, 15]])
        model.metrics_handler.val_confusion.compute = MagicMock(return_value=mock_cm)
        model.metrics_handler.val_confusion.reset = MagicMock()

        model.on_validation_epoch_end()

        # Check logs
        model.log.assert_any_call("val_cm_tn", 10, on_epoch=True, sync_dist=True)
        model.log.assert_any_call("val_cm_fp", 2, on_epoch=True, sync_dist=True)
        model.log.assert_any_call("val_cm_fn", 3, on_epoch=True, sync_dist=True)
        model.log.assert_any_call("val_cm_tp", 15, on_epoch=True, sync_dist=True)

        # Check reset
        model.metrics_handler.val_confusion.reset.assert_called_once()

    # --- 2. Loss Function Selection and Calculation ---

    def test_loss_selection_logic(self, mock_config, mock_resnet_head):
        """Regression: Ensure correct loss function is selected based on flags."""
        # Case 1: Focal + Smoothing
        model = LitResNetEnhanced(
            config=mock_config, use_focal_loss=True, label_smoothing=0.1
        )
        assert isinstance(model.loss_factory, FocalLossWithLabelSmoothing)

        # Case 2: Focal only
        model = LitResNetEnhanced(
            config=mock_config, use_focal_loss=True, label_smoothing=0
        )
        assert isinstance(model.loss_factory, FocalLoss)

        # Case 3: BCE
        model = LitResNetEnhanced(config=mock_config, use_focal_loss=False)
        assert isinstance(model.loss_factory, nn.BCEWithLogitsLoss)

    def test_loss_with_class_weights(self, mock_config, mock_resnet_head):
        """Regression: Ensure class weights are correctly converted to pos_weight."""
        weights = torch.tensor([1.0, 4.0])  # 4x more weight on positive class
        model = LitResNetEnhanced(
            config=mock_config, use_focal_loss=False, class_weights_tensor=weights
        )

        # pos_weight = 4.0 / 1.0 = 4.0
        assert torch.allclose(model.loss_factory.pos_weight, torch.tensor([4.0]))

    def test_loss_calculation_reshaping(self, mock_config, mock_resnet_head):
        """Regression: Ensure targets are reshaped correctly for loss calculation."""
        model = LitResNetEnhanced(config=mock_config)
        logits = torch.randn(4, 1)
        targets = torch.tensor([0, 1, 0, 1])

        # Patch the loss_factory's forward/call instead of the attribute itself
        with patch.object(
            model.loss_factory, "forward", return_value=torch.tensor(0.5)
        ) as mock_loss_forward:
            model._calculate_loss(logits, targets)
            mock_loss_forward.assert_called_once()
            args, _ = mock_loss_forward.call_args
            assert args[1].shape == (4, 1)
            assert args[1].dtype == torch.float32

    # --- 3. Optimizer and Scheduler Configuration ---

    def test_optimizer_config(self, mock_config, mock_resnet_head):
        """Regression: Ensure AdamW is configured with correct LR and WD."""
        model = LitResNetEnhanced(config=mock_config)

        # Mock parameters to avoid "empty parameter list" error
        with patch.object(
            model, "parameters", return_value=[nn.Parameter(torch.randn(1, 1))]
        ):
            opt_config = model.configure_optimizers()
            optimizer = opt_config["optimizer"]

            assert isinstance(optimizer, torch.optim.AdamW)
            assert optimizer.param_groups[0]["lr"] == 0.001
            assert optimizer.param_groups[0]["weight_decay"] == 0.0001

    def test_cosine_scheduler_config(self, mock_config, mock_resnet_head):
        """Regression: Ensure CosineAnnealingWarmRestarts parameters are correct."""
        model = LitResNetEnhanced(config=mock_config, use_cosine_scheduler=True)

        with patch.object(
            model, "parameters", return_value=[nn.Parameter(torch.randn(1, 1))]
        ):
            opt_config = model.configure_optimizers()
            scheduler = opt_config["lr_scheduler"]["scheduler"]

            assert isinstance(
                scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
            )
            assert scheduler.T_0 == 10 // 3
            assert scheduler.eta_min == 1e-7

    def test_plateau_scheduler_config(self, mock_config, mock_resnet_head):
        """Regression: Ensure ReduceLROnPlateau parameters are correct."""
        model = LitResNetEnhanced(
            config=mock_config, use_cosine_scheduler=False, monitor_metric="val_recall"
        )

        with patch.object(
            model, "parameters", return_value=[nn.Parameter(torch.randn(1, 1))]
        ):
            opt_config = model.configure_optimizers()
            scheduler = opt_config["lr_scheduler"]["scheduler"]

            assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
            assert opt_config["lr_scheduler"]["monitor"] == "val_recall"
            assert scheduler.mode == "max"  # because 'recall' is in monitor_metric
            assert scheduler.factor == 0.5
            assert scheduler.patience == 3

    # --- 4. Progressive Unfreezing Logic ---

    def test_progressive_unfreeze_interaction(self, mock_config, mock_resnet_head):
        """Regression: Ensure unfreezing logic correctly interacts with the model."""
        model = LitResNetEnhanced(config=mock_config)

        model.progressive_unfreeze(layers_to_unfreeze=2)
        assert model.unfrozen_layers == 2
        model.model._unfreeze_last_n_layers.assert_called_with(2)

        model.progressive_unfreeze(layers_to_unfreeze=3)
        assert model.unfrozen_layers == 5
        model.model._unfreeze_last_n_layers.assert_called_with(5)

    def test_freeze_resets_counter(self, mock_config, mock_resnet_head):
        """Regression: Ensure freeze_backbone resets the unfreeze counter."""
        model = LitResNetEnhanced(config=mock_config)
        model.unfrozen_layers = 10

        model.freeze_backbone()
        assert model.unfrozen_layers == 0
        model.model.freeze_backbone.assert_called_once()
