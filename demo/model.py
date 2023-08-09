# coding: utf-8

"""This file handles the pytorch lightning model and its training."""

import lightning as pl
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from demo.metrics import dice_loss, bce_dice_loss


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

    def forward(self, x):
        # Encoder
        conv1 = self.conv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.conv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.conv_down3(x)
        x = self.maxpool(conv3)
        conv4 = self.conv_down4(x)
        x = self.maxpool(conv4)
        x = self.conv_down5(x)

        # Decoder
        x = self.upsampler(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.conv_up4(x)
        x = self.upsampler(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv_up3(x)
        x = self.upsampler(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv_up2(x)
        x = self.upsampler(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv_up1(x)
        x = self.dropout(x)

        x = self.last_conv(x)
        x = torch.sigmoid(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        dice_loss_ = dice_loss(y_hat, y.unsqueeze(1))
        dsc = 1 - dice_loss_
        loss = bce_dice_loss(y_hat, y.unsqueeze(1))
        self.log("train_dsc", dsc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        dice_loss_ = dice_loss(y_hat, y.unsqueeze(1))
        dsc = 1 - dice_loss_
        loss = bce_dice_loss(y_hat, y.unsqueeze(1))
        self.log("val_dsc", dsc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        y_hat = self(x)
        dice_loss_ = dice_loss(y_hat, y.unsqueeze(1))
        dsc = 1 - dice_loss_
        loss = bce_dice_loss(y_hat, y.unsqueeze(1))
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
    trainer.save_checkpoint("model.ckpt")
    export(model, torch.zeros((1, 3, 256, 256)), "model_trained.onnx", verbose=True)
