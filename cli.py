from lightning.pytorch.cli import LightningCLI
from dataset import ShapesDataModule
from experiment import LightningVAE


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("offline_run", default=False)

    def before_instantiate_classes(self) -> None:
        super().before_instantiate_classes()
        if self.config[self.subcommand].offline_run:
            self.config[self.subcommand].trainer.logger = None


def cli_main():
    cli = MyLightningCLI(
        LightningVAE,
        ShapesDataModule,
        seed_everything_default=True,
        save_config_callback=None,
    )


if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block
