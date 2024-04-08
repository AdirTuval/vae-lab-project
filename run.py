from arg_parser import parse_config
from experiment import LightningVAE
from models import VanillaVAE
from lightning.pytorch import Trainer, seed_everything
from dataset import ShapesDataModule
from lightning.pytorch.loggers import WandbLogger

if __name__ == "__main__":
    config = parse_config()
    seed_everything(config['manual_seed'], workers=True)
    model = LightningVAE(VanillaVAE(**config['architecture_params']), params=config['opt_params'])
    data = ShapesDataModule(**config['data_params'])
    data.setup()

    if config['logging_params']['name'] == 'wandb':
        wandb_logger = WandbLogger(**config['logging_params'])
    else:
        wandb_logger = None
        
    trainer = Trainer(**config['trainer_params'], logger=wandb_logger)
    trainer.fit(model, datamodule=data)