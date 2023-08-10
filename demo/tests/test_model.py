import lightning as pl
import torch
from torch.utils.data import DataLoader

from demo.dataset import RandomTensorDataset, TCIADataset
from demo.model import UNet
from demo.pipeline import train_transformer, val_transformer


def test_forward():
    model = UNet(n_classes=1, learning_rate=0.001)
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    assert output.shape == (1, 1, 256, 256)


def test_backward():
    model = UNet(n_classes=1, learning_rate=0.001)
    dataset = RandomTensorDataset(size=5)
    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=5,
        log_every_n_steps=5,  # log every 5 steps to avoid warning when testing
    )
    trainer.fit(
        model=model,
        train_dataloaders=DataLoader(dataset),
        val_dataloaders=DataLoader(dataset),
    )
    trainer.test(
        model=model, 
        dataloaders=DataLoader(dataset))
    assert trainer.logged_metrics["test_loss"] < 100


def test_TCIADataset():
    nb_images = 5
    TCIADataset(
        [torch.rand(3, 256, 256)] * nb_images, [torch.rand(1, 256, 256)] * nb_images
    )
    TCIADataset(
        [torch.rand(3, 256, 256)] * nb_images,
        [torch.rand(1, 256, 256)] * nb_images,
        transform=train_transformer(),
    )
    TCIADataset(
        [torch.rand(3, 256, 256)] * nb_images,
        [torch.rand(1, 256, 256)] * nb_images,
        transform=val_transformer(),
    )

if __name__ == "__main__":
    test_backward()