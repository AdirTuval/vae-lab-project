from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset import ShapesDataModule
from experiment import LightningVAE


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.add_argument("offline_run", default=False)
        # parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        # parser.set_defaults({"my_early_stopping.monitor": "val_loss", "my_early_stopping.patience": 5})

    def before_instantiate_classes(self) -> None:
        super().before_instantiate_classes()
        if self.config[self.subcommand].offline_run:
            self.config[self.subcommand].trainer.logger = None


def cli_main():
    cli = MyLightningCLI(
        LightningVAE,
        ShapesDataModule,
        seed_everything_default=True,
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
