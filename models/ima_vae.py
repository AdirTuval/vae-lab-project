import torch
from torch import nn
from torch.nn import functional as F
from torch import tensor as Tensor
from torch.func import jacfwd, jacrev, vmap
from metrics import calculate_mcc
import numpy as np


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class IMA_Vae(nn.Module):

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_dims: list,
        decoder_var: float,
        **kwargs
    ) -> None:
        super(IMA_Vae, self).__init__()
        self.latent_dim = latent_dim
        self.latent_mapping = None
        self.decoder_var = decoder_var
        self.init_nets(in_channels, latent_dim, hidden_dims)

    def init_nets(self, in_channels: int, latent_dim: int, hidden_dims: list):
        self.last_hidden_dim = hidden_dims[-1]
        print("Hidden dims: ", hidden_dims)
        self.init_encoder2(in_channels, latent_dim, hidden_dims)
        self.init_decoder2(in_channels, latent_dim, hidden_dims)

    def init_encoder(self, in_channels: int, latent_dim: int, hidden_dims: list):
        modules = []

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(
            hidden_dims[-1] * 4, latent_dim
        )  # 4 beacuse of the we assume the shape now is hidden_dims[-1] x 2 x 2
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

    def init_decoder(self, in_channels: int, latent_dim: int, hidden_dims: list):
        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims = hidden_dims[::-1]

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                hidden_dims[-1], out_channels=in_channels, kernel_size=3, padding=1
            ),
            nn.Tanh(),
        )

    def init_encoder2(self, in_channels: int, latent_dim: int, hidden_dims: list):
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(),
            View((-1, 32 * 4 * 4)),
            nn.Linear(32 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2),
        )

    def init_decoder2(self, in_channels: int, latent_dim: int, hidden_dims: list):
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 32 * 4 * 4),
            nn.ReLU(),
            View((-1, 32, 4, 4)),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 4, 2, 1),
        )

    def encode(self, input: Tensor) -> list[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]
    
    def encode2(self, input: Tensor) -> list[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)

        # Split the result into mu and var components
        # of the latent Gaussian distribution

        mu = result[:, : self.latent_dim]
        log_var = result[:, self.latent_dim:]

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        if self.latent_mapping is not None:
            print("Using latent mapping")
            z = self.latent_mapping(z)
        result = self.decoder(z)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> list[Tensor]:
        mu, log_var = self.encode2(input)
        z = self.reparameterize(mu, log_var)
        return {
            "recons": self.decode(z),
            "mu": mu,
            "log_var": log_var,
            "input": input,
            "latents": z,
        }

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = kwargs["recons"]
        mu = kwargs["mu"]
        log_var = kwargs["log_var"]
        input = kwargs["input"]

        kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input) / self.decoder_var

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        loss = recons_loss + kld_weight * kld_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": -kld_loss.detach(),
        }

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    def calculate_decoder_jacobian(self, z: Tensor):
        """
        Calculates the Jacobian of the decoder w.r.t. the latent code z.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x D x (C*H*W)]
        """

        # Flatten decoder:
        def flatten_decoder(x):
            return self.decode(x).flatten()

        return vmap(jacfwd(flatten_decoder))(z)

    def learn_latent_mapping(self, sources, samples) -> Tensor:
        latents = self.forward(samples)["latents"]
        latents = latents.detach().cpu().numpy()
        sources = sources.detach().cpu().numpy()
        a0, b0 = np.polyfit(sources[0, :], latents[0, :], 1)
        a1, b1 = np.polyfit(sources[1, :], latents[1, :], 1)
        a = torch.Tensor([a0, a1]).cuda()
        b = torch.Tensor([b0, b1]).cuda()
        self.latent_mapping = lambda t: t * a + b
