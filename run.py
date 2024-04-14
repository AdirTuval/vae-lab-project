from arg_parser import parse_config
from experiment import LightningVAE
from models import VanillaVAE
from lightning.pytorch import Trainer, seed_everything
from dataset import ShapesDataModule
from lightning.pytorch.loggers import WandbLogger

if __name__ == "__main__":
    model = LightningVAE.load_from_checkpoint("/cs/labs/yweiss/adirt/lab_project/vae-lab-project/lightning_logs/h2sycu3v/checkpoints/epoch=99-step=12500.ckpt")
    # config = parse_config()
    # seed_everything(config['manual_seed'], workers=True)
    # # model = LightningVAE(VanillaVAE(**config['architecture_params']), params=config['opt_params'])
    # model = LightningVAE(**config)
    # data = ShapesDataModule(**config['data_params'])
    # data.setup()

    # if config['logging_params']['name'] == 'wandb':
    #     wandb_logger = WandbLogger(**config['logging_params'])
    # else:
    #     wandb_logger = None
        
    # trainer = Trainer(**config['trainer_params'], logger=wandb_logger)
    # trainer.fit(model, datamodule=data)