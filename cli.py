from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from dataset import ShapesDataModule
from experiment import LightningVAE
import torch

torch.multiprocessing.set_sharing_strategy("file_system")


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.add_argument("offline_run", default=False)
        parser.add_lightning_class_args(EarlyStopping, "early_stopping")


def cli_main():
    cli = MyLightningCLI(
        LightningVAE,
        ShapesDataModule,
        seed_everything_default=True,
        save_config_callback=None,
        trainer_defaults={
            "callbacks": [
                ModelCheckpoint(
                    monitor="Validation/ELBO_Loss", save_top_k=1, mode="min"
                )
            ]
        },
    )


if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block
