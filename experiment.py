import lightning as L
from models import BaseVAE
from torch import optim

class LightningVAE(L.LightningModule):
    def __init__(self, model: BaseVAE):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
