from experiment import LightningVAE
from models import VanillaVAE
import lightning as L
from shapes_dataset_generator import ShapesDatasetGenerator

if __name__ == "__main__":
    print("hello")
    model = LightningVAE(VanillaVAE(in_channels=3, latent_dim=128))
    dataset, _ = ShapesDatasetGenerator(
        random_seed=42, render_config=None
    ).generate(n_samples=10000)
    trainer = L.Trainer(max_epochs=1, limit_train_batches=100)
    trainer.fit(model, dataset)