# coding: utf-8

"""This file handles the pytorch lightning model and its training."""

import lightning as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from demo.metrics import bce_dice_loss, dice_loss


class UNet(pl.LightningModule):
    def __init__(self, n_classes, learning_rate):
        super().__init__()

        # Save inputs to hparams
        self.save_hyperparameters()

        self.learning_rate = learning_rate

        # Convolutional layers
        self.conv_down1 = self._double_conv(3, 32)
        self.conv_down2 = self._double_conv(32, 64)
        self.conv_down3 = self._double_conv(64, 128)
        self.conv_down4 = self._double_conv(128, 256)
        self.conv_down5 = self._double_conv(256, 512)

        # Upsampling layers
        self.upsampler = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

        self.conv_up4 = self._double_conv(256 + 512, 256)
        self.conv_up3 = self._double_conv(128 + 256, 128)
        self.conv_up2 = self._double_conv(64 + 128, 64)
        self.conv_up1 = self._double_conv(32 + 64, 32)

        self.last_conv = nn.Conv2d(32, n_classes, kernel_size=1)

        self.dropout = nn.Dropout2d()

        # Activation functions
        self.maxpool = nn.MaxPool2d(2)

    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.ReLU(inplace=True),
        )

    def configure_callbacks(self):
        return [
            EarlyStopping(monitor="val_loss", mode="min", patience=10),
            LearningRateMonitor("epoch"),
        ]

    def forward(self, input_tensor):
        # Encoder
        conv1 = self.conv_down1(input_tensor)
        transformed_tensor = self.maxpool(conv1)
        conv2 = self.conv_down2(transformed_tensor)
        transformed_tensor = self.maxpool(conv2)
        conv3 = self.conv_down3(transformed_tensor)
        transformed_tensor = self.maxpool(conv3)
        conv4 = self.conv_down4(transformed_tensor)
        transformed_tensor = self.maxpool(conv4)
        transformed_tensor = self.conv_down5(transformed_tensor)

        # Decoder
        transformed_tensor = self.upsampler(transformed_tensor)
        transformed_tensor = torch.cat([transformed_tensor, conv4], dim=1)
        transformed_tensor = self.conv_up4(transformed_tensor)
        transformed_tensor = self.upsampler(transformed_tensor)
        transformed_tensor = torch.cat([transformed_tensor, conv3], dim=1)
        transformed_tensor = self.conv_up3(transformed_tensor)
        transformed_tensor = self.upsampler(transformed_tensor)
        transformed_tensor = torch.cat([transformed_tensor, conv2], dim=1)
        transformed_tensor = self.conv_up2(transformed_tensor)
        transformed_tensor = self.upsampler(transformed_tensor)
        transformed_tensor = torch.cat([transformed_tensor, conv1], dim=1)
        transformed_tensor = self.conv_up1(transformed_tensor)
        transformed_tensor = self.dropout(transformed_tensor)

        transformed_tensor = self.last_conv(transformed_tensor)
        transformed_tensor = torch.sigmoid(transformed_tensor)

        return transformed_tensor

    def training_step(self, batch, batch_idx):
        input_tensor, mask_tensor = batch
        y_hat = self(input_tensor)
        dice_loss_ = dice_loss(y_hat, mask_tensor.unsqueeze(1))
        dsc = 1 - dice_loss_
        loss = bce_dice_loss(y_hat, mask_tensor.unsqueeze(1))
        self.log("train_dsc", dsc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_tensor, mask_tensor = batch
        y_hat = self(input_tensor)
        dice_loss_ = dice_loss(y_hat, mask_tensor.unsqueeze(1))
        dsc = 1 - dice_loss_
        loss = bce_dice_loss(y_hat, mask_tensor.unsqueeze(1))
        self.log("val_dsc", dsc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        input_tensor, mask_tensor = batch
        y_hat = self(input_tensor)
        dice_loss_ = dice_loss(y_hat, mask_tensor.unsqueeze(1))
        dsc = 1 - dice_loss_
        loss = bce_dice_loss(y_hat, mask_tensor.unsqueeze(1))
        self.log("test_dsc", dsc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adamax(self.parameters(), lr=self.learning_rate)
        lr_scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer, patience=5),
            "name": "ReduceLROnPlateau_log",
            "monitor": "val_loss",
        }
        return [optimizer], [lr_scheduler]
