from torch import optim
from models import VanillaVAE
from torch import tensor as Tensor
import lightning as L
from metrics.mcc import mcc


class LightningVAE(L.LightningModule):

    def __init__(self, vae_model: VanillaVAE, params: dict) -> None:
        super(LightningVAE, self).__init__()
        self.model = vae_model
        self.params = params
        self.save_hyperparameters(ignore=['vae_model'])

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        samples, sources = batch
        results = self.forward(samples)
        train_loss = self.model.loss_function(
            **results,
            M_N=self.params["kld_weight"],  # al_img.shape[0]/ self.num_train_imgs,
        )

        # Log 
        self.log("Train/ELBO_Loss", train_loss["loss"])
        self.log("Train/Reconstruction_Loss", train_loss["Reconstruction_Loss"])
        self.log("Train/Mean_Correlation_Coefficient", self._calculate_mcc(results['latents'], sources))

        return train_loss["loss"]

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        pass
        # real_img, labels = batch
        # self.curr_device = real_img.device

        # results = self.forward(real_img, labels=labels)
        # val_loss = self.model.loss_function(
        #     *results,
        #     M_N=1.0,  # real_img.shape[0]/ self.num_val_imgs,
        #     optimizer_idx=optimizer_idx,
        #     batch_idx=batch_idx
        # )

        # self.log_dict(
        #     {f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True
        # )

    # def on_validation_end(self) -> None:
    #     self.sample_images()

    # def sample_images(self):
    #     # Get sample reconstruction image
    #     test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
    #     test_input = test_input.to(self.curr_device)
    #     test_label = test_label.to(self.curr_device)

    #     #         test_input, test_label = batch
    #     recons = self.model.generate(test_input, labels=test_label)
    #     vutils.save_image(
    #         recons.data,
    #         os.path.join(
    #             self.logger.log_dir,
    #             "Reconstructions",
    #             f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png",
    #         ),
    #         normalize=True,
    #         nrow=12,
    #     )

    #     try:
    #         samples = self.model.sample(144, self.curr_device, labels=test_label)
    #         vutils.save_image(
    #             samples.cpu().data,
    #             os.path.join(
    #                 self.logger.log_dir,
    #                 "Samples",
    #                 f"{self.logger.name}_Epoch_{self.current_epoch}.png",
    #             ),
    #             normalize=True,
    #             nrow=12,
    #         )
    #     except Warning:
    #         pass

    def _calculate_mcc(self, latents, sources):
        latents_copy = latents.detach().permute(1, 0).numpy()
        sources_copy = sources.detach().permute(1, 0).numpy()
        mcc_for_batch, _, _, _ = mcc(latents_copy, sources_copy)
        return mcc_for_batch

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.params["LR"],
            weight_decay=self.params["weight_decay"],
        )
        optims.append(optimizer)

        scheduler = optim.lr_scheduler.ExponentialLR(
            optims[0], gamma=self.params["scheduler_gamma"]
        )
        scheds.append(scheduler)

        return optims, scheds
