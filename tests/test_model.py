from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from demo.model import UNet

def test_forward():
    model = UNet(n_classes=1, learning_rate=0.001)
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    assert output.shape == (1, 1, 256, 256)


class RandomTensorDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.randn(3, 256, 256)
        y = torch.randn(1, 256, 256)
        return x, y
        
class ModelTest(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = UNet(n_classes=1, learning_rate=0.001)
        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)
        self.log('train_loss', loss)
        return loss

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = RandomTensorDataset(size=15)
        return DataLoader(dataset)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

def test_backward():
    model = ModelTest()
    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(model)
    assert trainer.logged_metrics['train_loss'] < 100
    