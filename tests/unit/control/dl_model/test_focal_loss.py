import pytest
import torch
import torch.nn as nn
from federated_pneumonia_detection.src.control.dl_model.internals.model.losses.focal_loss import (
    FocalLoss,
    FocalLossWithLabelSmoothing,
)


def test_focal_loss_initialization():
    """Test FocalLoss initialization with custom parameters."""
    alpha = 0.5
    gamma = 3.0
    pos_weight = torch.tensor([2.0])
    reduction = "sum"

    loss_fn = FocalLoss(
        alpha=alpha, gamma=gamma, pos_weight=pos_weight, reduction=reduction
    )

    assert loss_fn.alpha == alpha
    assert loss_fn.gamma == gamma
    assert loss_fn.pos_weight is not None
    assert torch.equal(loss_fn.pos_weight, pos_weight)
    assert loss_fn.reduction == reduction


def test_focal_loss_perfect_predictions():
    """Test that perfect predictions result in very low loss."""
    loss_fn = FocalLoss(gamma=2.0, alpha=0.25)

    # logits: high for positive, low for negative
    inputs = torch.tensor([10.0, -10.0])
    targets = torch.tensor([1.0, 0.0])

    loss = loss_fn(inputs, targets)

    # Loss should be near 0
    assert loss.item() < 1e-4


def test_focal_loss_wrong_predictions():
    """Test that wrong predictions result in high loss."""
    loss_fn = FocalLoss(gamma=2.0, alpha=0.25)

    # logits: low for positive, high for negative
    inputs = torch.tensor([-10.0, 10.0])
    targets = torch.tensor([1.0, 0.0])

    loss = loss_fn(inputs, targets)

    # Loss should be high
    assert loss.item() > 1.0


def test_focal_loss_alpha_weighting():
    """Test that alpha correctly weights positive and negative classes."""
    # alpha=0.8 means positive class is weighted 0.8, negative 0.2
    loss_fn_alpha = FocalLoss(alpha=0.8, gamma=0.0, reduction="none")
    loss_fn_no_alpha = FocalLoss(alpha=-1, gamma=0.0, reduction="none")

    inputs = torch.tensor([0.0, 0.0])  # p=0.5
    targets = torch.tensor([1.0, 0.0])

    loss_alpha = loss_fn_alpha(inputs, targets)
    loss_no_alpha = loss_fn_no_alpha(inputs, targets)

    # For positive class (target=1.0): loss_alpha = 0.8 * loss_no_alpha
    assert torch.allclose(loss_alpha[0], 0.8 * loss_no_alpha[0])
    # For negative class (target=0.0): loss_alpha = (1-0.8) * loss_no_alpha
    assert torch.allclose(loss_alpha[1], 0.2 * loss_no_alpha[1])


def test_focal_loss_gamma_focusing():
    """Test that higher gamma reduces loss more for easy examples."""
    # Easy example: logit=2.0 (p~0.88), target=1.0
    inputs = torch.tensor([2.0])
    targets = torch.tensor([1.0])

    loss_gamma_0 = FocalLoss(gamma=0.0, alpha=-1)(inputs, targets)
    loss_gamma_2 = FocalLoss(gamma=2.0, alpha=-1)(inputs, targets)
    loss_gamma_5 = FocalLoss(gamma=5.0, alpha=-1)(inputs, targets)

    # Higher gamma should result in lower loss for easy examples
    assert loss_gamma_2 < loss_gamma_0
    assert loss_gamma_5 < loss_gamma_2


def test_focal_loss_reduction_modes():
    """Test different reduction modes."""
    inputs = torch.tensor([1.0, -1.0])
    targets = torch.tensor([1.0, 0.0])

    # Mean
    loss_mean = FocalLoss(reduction="mean")(inputs, targets)
    # Sum
    loss_sum = FocalLoss(reduction="sum")(inputs, targets)
    # None
    loss_none = FocalLoss(reduction="none")(inputs, targets)

    assert loss_none.shape == (2,)
    assert torch.allclose(loss_mean, loss_none.mean())
    assert torch.allclose(loss_sum, loss_none.sum())


def test_focal_loss_shapes():
    """Test that (N, 1) and (N,) shapes are handled correctly."""
    loss_fn = FocalLoss()

    inputs_1d = torch.tensor([1.0, -1.0])
    targets_1d = torch.tensor([1.0, 0.0])

    inputs_2d = torch.tensor([[1.0], [-1.0]])
    targets_2d = torch.tensor([[1.0], [0.0]])

    loss_1d = loss_fn(inputs_1d, targets_1d)
    loss_2d = loss_fn(inputs_2d, targets_2d)

    assert torch.allclose(loss_1d, loss_2d)


def test_focal_loss_with_label_smoothing_initialization():
    """Test FocalLossWithLabelSmoothing initialization and validation."""
    loss_fn = FocalLossWithLabelSmoothing(smoothing=0.2)
    assert loss_fn.smoothing == 0.2

    with pytest.raises(ValueError, match=r"Smoothing must be in \[0, 0.5\)"):
        FocalLossWithLabelSmoothing(smoothing=0.6)


def test_focal_loss_with_label_smoothing_forward():
    """Test FocalLossWithLabelSmoothing forward pass logic."""
    smoothing = 0.1
    loss_fn = FocalLossWithLabelSmoothing(smoothing=smoothing, gamma=2.0, alpha=0.25)

    # Use logits that are not too large to see the effect of smoothing
    inputs = torch.tensor([1.0, -1.0])
    targets = torch.tensor([1.0, 0.0])

    # Should run without error
    loss = loss_fn(inputs, targets)
    assert isinstance(loss, torch.Tensor)

    # Compare with standard FocalLoss
    loss_standard = FocalLoss(gamma=2.0, alpha=0.25)(inputs, targets)

    # With smoothing, the loss should be different and typically higher for these inputs
    # because the targets are moved away from 0/1
    assert not torch.allclose(loss, loss_standard, atol=1e-6)
    assert loss > loss_standard


def test_focal_loss_pos_weight():
    """Test that pos_weight correctly weights the positive class."""
    pos_weight = torch.tensor([5.0])
    loss_fn = FocalLoss(pos_weight=pos_weight, alpha=-1, gamma=0.0, reduction="none")
    loss_fn_no_weight = FocalLoss(
        pos_weight=None, alpha=-1, gamma=0.0, reduction="none"
    )

    inputs = torch.tensor([0.0, 0.0])
    targets = torch.tensor([1.0, 0.0])

    loss_weighted = loss_fn(inputs, targets)
    loss_unweighted = loss_fn_no_weight(inputs, targets)

    # Positive class should be weighted by 5.0
    assert torch.allclose(loss_weighted[0], 5.0 * loss_unweighted[0])
    # Negative class should be weighted by 1.0
    assert torch.allclose(loss_weighted[1], loss_unweighted[1])
