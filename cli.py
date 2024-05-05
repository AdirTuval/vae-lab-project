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


def cli_main():
    cli = MyLightningCLI(
        LightningVAE,
        ShapesDataModule,
        save_config_callback=None,
        trainer_defaults={
            "callbacks": [
                ModelCheckpoint(
                    monitor="Validation/ELBO_Loss", save_top_k=1, mode="min"
                )
            ],
            "deterministic" : True
            
        },
    )


if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block
