import torch
from typing import Optional
import numpy as np
import lightning as L
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from shapes_dataset_generator import ShapesDatasetGenerator


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
        generate_data_on_the_fly: bool,
        existing_data_path: dict,
        data_generation_params: dict,
        general_data_params: dict,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        if self.hparams.generate_data_on_the_fly:
            print("Generating data on the fly...")
            self.shapes, self.sources = ShapesDatasetGenerator(
                self.hparams.data_generation_params["render_config"], 
                self.hparams.data_generation_params["sampler_config"]
            ).generate(self.hparams.data_generation_params["n_samples"])
            print("Data generated successfully!")
        else:
            shapes_path = self.hparams.existing_data_path["shapes"]
            sources_path = self.hparams.existing_data_path["sources"]
            self.shapes = np.load(shapes_path)
            self.sources = np.load(sources_path)

        dataset = ShapesDataset(shapes=self.shapes, sources=self.sources)

        # split
        train_len = int(self.hparams.general_data_params["train_ratio"] * len(dataset))
        val_len = int(self.hparams.general_data_params["val_ratio"] * len(dataset))

        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_len, val_len]
        )
        if self.hparams.general_data_params["save_data"]:
            np.save("/cs/labs/yweiss/adirt/lab_project/vae-lab-project/data/shapes.npy", self.shapes)
            np.save("/cs/labs/yweiss/adirt/lab_project/vae-lab-project/data/sources.npy", self.sources)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.general_data_params["train_batch_size"],
            num_workers=self.hparams.general_data_params["num_workers"],
            shuffle=self.hparams.general_data_params["train_batch_shuffle"]
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.general_data_params["train_batch_size"],
            num_workers=self.hparams.general_data_params["num_workers"],
        )
