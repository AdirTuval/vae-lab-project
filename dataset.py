import torch
from typing import Union, Optional
import numpy as np
import lightning as L
from torch.utils.data import DataLoader, Dataset
from shapes_dataset_generator import ShapesDatasetGenerator


class ShapesDataset(Dataset):
    """
    A PyTorch Dataset class for the shapes dataset.
    """

    def __init__(self, shapes_path: str, sources_path: str, split: str, transform=None):
        assert split in ["train", "val", "test"]
        self.transforms = transform
        self.sources = np.load(sources_path)

        # Set right order of dimensions
        self.shapes = np.load(shapes_path)
        self.shapes = np.transpose(self.shapes, (0, 3, 1, 2))

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
        **kwargs
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.shapes_path = shapes_path
        self.sources_path = sources_path
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = ShapesDataset(
            shapes_path=self.shapes_path,
            sources_path=self.sources_path,
            split="train",
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
        )
