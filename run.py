from arg_parser import parse_config
from experiment import LightningVAE
from models import VanillaVAE
from lightning.pytorch import Trainer, seed_everything
from dataset import ShapesDataModule
if __name__ == "__main__":
    config = parse_config()
    seed_everything(config['manual_seed'], workers=True)
    model = LightningVAE(VanillaVAE(**config['architecture_params']), params=config['opt_params'])
    data = ShapesDataModule(**config['data_params'])
    data.setup()

    trainer = Trainer(**config['trainer_params'])
    trainer.fit(model, datamodule=data)