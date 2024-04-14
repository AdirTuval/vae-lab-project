import torch
from typing import Optional
import numpy as np
import lightning as L
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split

class ShapesDataset(Dataset):
    """
    A PyTorch Dataset class for the shapes dataset.
    """

    def __init__(self, shapes: np.array, sources: np.array, transform=None):
        self.transforms = transform
        self.sources = sources
        # Set right order of dimensions
        self.shapes = np.transpose(shapes, (0, 3, 1, 2))

    def __len__(self):
        return len(self.shapes)

    def __getitem__(self, idx):
        sample = self.shapes[idx]
        source = self.sources[idx]

        # Convert the sample to PyTorch tensor if needed
        sample = torch.from_numpy(sample).float() / 255.0

        # Apply additional transformations if specified
        if self.transforms:
            sample = self.transforms(sample)

        return sample, source


class ShapesDataModule(L.LightningDataModule):
    """
    A PyTorch Lightning DataModule for the shapes dataset.
    """

    def __init__(
        self,
        shapes_path: str,
        sources_path: str,
        train_batch_size: int = 8,
        num_workers: int = 4,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.shapes_path = shapes_path
        self.sources_path = sources_path
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

    def setup(self, stage: Optional[str] = None) -> None:
        # Load shapes and sources
        self.shapes = np.load(self.shapes_path)
        self.sources = np.load(self.sources_path)
        dataset = ShapesDataset(shapes=self.shapes, sources=self.sources)
        
        # split
        train_len = int(self.train_ratio * len(dataset))
        val_len = int(self.val_ratio * len(dataset))

        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_len, val_len]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
        )
