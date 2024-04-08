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

    def __init__(
        self, data_path: Union[str, None], split: str, transform=None, generate=False
    ):
        assert split in ["train", "val", "test"]
        assert data_path is not None or generate
        self.transforms = transform
        if generate:
            self.data = ShapesDatasetGenerator().generate(n_samples=10000)
        if data_path is not None:
            self.data = np.load(data_path)

        # Set right order of dimensions
        self.data = np.transpose(self.data, (0, 3, 1, 2))    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Convert the sample to PyTorch tensor if needed
        sample = torch.from_numpy(sample).float() / 255.0

        # Apply additional transformations if specified
        if self.transforms:
            sample = self.transforms(sample)

        return sample


class ShapesDataModule(L.LightningDataModule):
    """
    A PyTorch Lightning DataModule for the shapes dataset.
    """

    def __init__(
        self,
        data_path: Union[str, None],
        train_batch_size: int = 8,
        generate: bool = False,
        num_workers: int = 4,
        **kwargs
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.data_path = data_path
        self.generate = generate
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = None
        self.train_dataset = ShapesDataset(
            data_path=self.data_path, split="train", generate=self.generate
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.num_workers)
