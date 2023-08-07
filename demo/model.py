# coding: utf-8

"""This file handles the pytorch lightning model and its training."""

import lightning as pl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.callbacks import Callback
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional.image import \
    multiscale_structural_similarity_index_measure


class UNet(pl.LightningModule):
    def __init__(self, n_classes, learning_rate):
        super().__init__()

        # Save inputs to hparams
        self.save_hyperparameters()

        self.learning_rate = learning_rate

        # Convolutional layers
        self.conv_down1 = self._double_conv(3, 64)
        self.conv_down2 = self._double_conv(64, 128)
        self.conv_down3 = self._double_conv(128, 256)
        self.conv_down4 = self._double_conv(256, 512)

        # Upsampling layers
        self.upsampler = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

        self.conv_up3 = self._double_conv(256 + 512, 256)
        self.conv_up2 = self._double_conv(128 + 256, 128)
        self.conv_up1 = self._double_conv(128 + 64, 64)

        self.last_conv = nn.Conv2d(64, n_classes, kernel_size=1)

        # Activation functions
        self.maxpool = nn.MaxPool2d(2)

    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _dice_loss(y_pred, y_true, smooth=1):
        # Flatten predictions and labels
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        # Calculate intersection and union
        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum()

        # Calculate Dice coefficient
        dice = (2.0 * intersection + smooth) / (union + smooth)

        # Calculate Dice loss
        loss = 1.0 - dice

        return loss

    def configure_callbacks(self):
        return [
            EarlyStopping(monitor="val_loss", mode="min", patience=10),
            LearningRateMonitor("epoch"),
        ]

    def forward(self, x):
        conv1 = self.conv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.conv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.conv_down3(x)
        x = self.maxpool(conv3)
        x = self.conv_down4(x)

        x = self.upsampler(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv_up3(x)
        x = self.upsampler(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv_up2(x)
        x = self.upsampler(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv_up1(x)

        x = self.last_conv(x)
        x = torch.sigmoid(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self._dice_loss(y_hat, y.unsqueeze(1))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self._dice_loss(y_hat, y.unsqueeze(1))
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        y_hat = self(x)
        loss = self._dice_loss(y_hat, y.unsqueeze(1))
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adamax(self.parameters(), lr=self.learning_rate)
        lr_scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer, patience=5),
            "name": "ReduceLROnPlateau_log",
            "monitor": "val_loss",
        }
        return [optimizer], [lr_scheduler]


if __name__ == "__main__":
    import pathlib

    from lightning.pytorch.callbacks.early_stopping import EarlyStopping
    from lightning.pytorch.tuner.tuning import Tuner
    from torch.onnx import export
    from torch.utils.data import DataLoader

    from demo.pipeline import pre_treatement_pipeline
    from demo.study import Study

    model = UNet(1, 0.001)

    export(model, torch.zeros((1, 3, 256, 256)), "model.onnx", verbose=True)

    study = Study(pathlib.Path(__file__).parent.parent / "data")

    train_dataset, validation_dataset, test_dataset = pre_treatement_pipeline(study)

    train_loader = DataLoader(train_dataset, num_workers=4)
    validation_loader = DataLoader(validation_dataset, num_workers=4)

    trainer = pl.Trainer(
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")], max_epochs=5
    )
    tuner = Tuner(trainer)

    # finds learning rate automatically
    # sets hparams.lr or hparams.learning_rate to that learning rate
    tuner.lr_find(
        model, train_dataloaders=train_loader, val_dataloaders=validation_loader
    )

    # Train the model
    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=validation_loader
    )

    # test the model
    trainer.test(model=model, dataloaders=DataLoader(test_dataset, num_workers=4))

    # save the model
    trainer.save_checkpoint("best_model.ckpt")
    export(model, torch.zeros((1, 3, 256, 256)), "model_trained.onnx", verbose=True)
