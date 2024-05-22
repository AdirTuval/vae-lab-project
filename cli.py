from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from dataset import ShapesDataModule
from experiment import LightningVAE
import torch

torch.multiprocessing.set_sharing_strategy("file_system")


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")

        def data_params_to_latent_dim(data_params):
            if data_params.generate_data_on_the_fly:
                return len(data_params.data_generation_params["factors"])
            # Infer from the data
            import numpy as np
            sources = np.load(data_params.existing_data_path["sources"])
            return sources.shape[-1]
        
        parser.link_arguments("data", "model.vae.init_args.latent_dim", data_params_to_latent_dim)

def cli_main():
    cli = MyLightningCLI(
        LightningVAE,
        ShapesDataModule,
        save_config_callback=None,
        trainer_defaults={
            "deterministic" : True  
        },
    )


if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block
