import numpy as np
import torch
from torch import optim
from models import VanillaVAE
from torch import tensor as Tensor
import lightning as L
from metrics.mcc import mcc


class LightningVAE(L.LightningModule):

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_dims: list,
        learning_rate: float,
        scheduler_gamma: float,
        kld_weight: float,
    ) -> None:
        super(LightningVAE, self).__init__()
        self.model = VanillaVAE(
            in_channels=in_channels, latent_dim=latent_dim, hidden_dims=hidden_dims
        )
        self.kld_weight = kld_weight
        self.learning_rate = learning_rate
        self.scheduler_gamma = scheduler_gamma
        self.validation_step_outputs = []
        self.latent_mapping = None

        self.save_hyperparameters()

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        samples, sources = batch
        results = self.forward(samples)
        train_loss = self.model.loss_function(
            **results,
            M_N=self.kld_weight,  # al_img.shape[0]/ self.num_train_imgs,
        )

        # Log
        self.log("Train/ELBO_Loss", train_loss["loss"], prog_bar=True)
        self.log("Train/Reconstruction_Loss", train_loss["Reconstruction_Loss"])
        self._log_mcc("Train", results["latents"], sources)
        return train_loss["loss"]

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        samples, sources = batch
        results = self.forward(samples)
        val_loss = self.model.loss_function(
            **results,
            M_N=self.kld_weight,  # al_img.shape[0]/ self.num_train_imgs,
        )

        # Log
        self.log("Validation/ELBO_Loss", val_loss["loss"])
        self.log("Validation/Reconstruction_Loss", val_loss["Reconstruction_Loss"])
        self._log_mcc("Validation", results["latents"], sources)

    def _log_mcc(self, prefix, latents, sources):
        curr_mcc, latents_sorted = self._calculate_mcc(latents, sources)
        self.log(f"{prefix}/Mean_Correlation_Coefficient", curr_mcc)

        if prefix == "Validation":
            self.validation_step_outputs.append(
                (
                    latents_sorted,
                    sources.detach().cpu().permute(1, 0).numpy(),
                )
            )

    def on_validation_epoch_end(self) -> None:
        self.learn_latent_mapping()
        images, latents = self.sample_base_images()
        self._log_images(images, latents)

    def learn_latent_mapping(self) -> Tensor:
        latents, sources = zip(*self.validation_step_outputs)
        latents = np.hstack(latents)
        sources = np.hstack(sources)
        a0, b0 = np.polyfit(sources[0, :], latents[0, :], 1)
        a1, b1 = np.polyfit(sources[1, :], latents[1, :], 1)
        self.latent_mapping = lambda t: t * np.array([a0, a1]) + np.array([b0, b1])
        self.validation_step_outputs = []

    def sample_base_images(self) -> list[Tensor]:
        mapped_latents, original_latents = self.get_base_latents()
        base_images = self.model.decode(mapped_latents)
        return base_images, original_latents

    def _log_images(self, images, latents):
        images = list(images.detach().cpu().permute(0, 2, 3, 1).numpy())
        self.logger.log_image(
            key="Validation/Latents_Samples",
            images=list(images),
            step=self.current_epoch,
            caption=["Latent: " + str(latent) for latent in latents],
        )

    def get_base_latents(self):
        x, y = torch.meshgrid(*(2 * [torch.arange(0, 1.001, 0.5)]), indexing="ij")
        base_latents = torch.stack((x.flatten(), y.flatten()), dim=-1)
        base_latents_after_mapping = self.latent_mapping(base_latents).to(self.device).type(torch.float32)
        return base_latents_after_mapping, base_latents

    def _calculate_mcc(self, latents, sources):
        latents_copy = latents.detach().cpu().permute(1, 0).numpy()
        sources_copy = sources.detach().cpu().permute(1, 0).numpy()
        corr_sort, _, latents_sorted = mcc(latents_copy, sources_copy)
        curr_mcc = np.mean(np.abs(np.diag(corr_sort)))
        return curr_mcc, latents_sorted

    def configure_optimizers(self):
        optims = []
        scheds = []

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            # weight_decay=self.params["weight_decay"],
        )
        optims.append(optimizer)

        scheduler = optim.lr_scheduler.ExponentialLR(
            optims[0], gamma=self.scheduler_gamma
        )
        scheds.append(scheduler)

        return optims, scheds
