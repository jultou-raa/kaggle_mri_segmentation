# coding: utf-8

"""This file handles the dataset classes"""

import torch
from torch.utils.data import Dataset


class RandomTensorDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.rand(3, 256, 256), torch.rand(256, 256)


class TCIADataset(Dataset):
    def __init__(self, images, masks, transform=None) -> None:
        super().__init__()
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask
