# coding: utf-8
"""This file handles the loss functions used in models"""

import torch
from torch import nn


def dice_loss(y_pred: torch.Tensor, y_true: torch.Tensor, smooth: float = 1):
    """Computes the DICE loss."""

    # Flatten predictions and labels
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)

    # Calculate intersection and union
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum()

    # Calculate Dice coefficient
    dice = (2.0 * intersection + smooth) / (union + smooth)

    # Return Dice loss
    return 1.0 - dice


def bce_dice_loss(y_pred: torch.Tensor, y_true: torch.Tensor, smooth: float = 1):
    """Compute the combined Binary Cross Entropy and DICE loss."""

    # BCE loss expected Float dtype input
    y_pred = y_pred.to(torch.float32, copy=True)
    y_true = y_true.to(torch.float32, copy=True)
    # Calculate BCE loss
    bce_loss = nn.BCELoss()(y_pred, y_true)

    # Calculate Dice loss
    dice_loss_ = dice_loss(y_pred, y_true, smooth=smooth)

    # Return combined loss
    return bce_loss + dice_loss_
