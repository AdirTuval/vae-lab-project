import numpy as np
import torch
from torch import distributions as dist

class Normal:
    """
    Code for Normal class adapted from: https://github.com/ilkhem/icebeem/blob/master/models/ivae/ivae_core.py
    """

    def __init__(self, diag=True):
        super().__init__()
        self.c = 2 * np.pi * torch.ones(1)
        self._dist = dist.normal.Normal(
            torch.zeros(1), torch.ones(1)
        )
        self.name = "gaussian"
        self.diag = True

    def sample(self, mu, diag=True):
        eps = self._dist.sample(mu.size()).squeeze()
        std = v.sqrt()
        if self.diag:
            scaled = eps.mul(std)
        else:
            # v is cholesky and not variance
            scaled = torch.matmul(v, eps.unsqueeze(2)).view(eps.shape)
        return scaled.add(mu)

    def log_pdf(self, x, mu, v):
        """compute the log-pdf of a normal distribution with diagonal covariance"""
        lpdf = -0.5 * (torch.log(self.c) + v.log() + (x - mu).pow(2).div(v))
        return lpdf.sum(dim=-1)

    def log_pdf_full(self, x, mu, v):
        """
        compute the log-pdf of a normal distribution with full covariance
        v is a batch of "pseudo sqrt" of covariance matrices of shape (batch_size, d_latent, d_latent)
        mu is batch of means of shape (batch_size, d_latent)
        """
        batch_size, d = mu.size()
        cov = torch.einsum(
            "bik,bjk->bij", v, v
        )  # compute batch cov from its "pseudo sqrt"
        assert cov.size() == (batch_size, d, d)
        inv_cov = torch.inverse(cov)
        c = d * torch.log(self.c)
        _, logabsdets = torch.slogdet(cov)
        xmu = x - mu
        lpdf = -0.5 * (
            c + logabsdets + torch.einsum("bi,bij,bj->b", [xmu, inv_cov, xmu])
        )
        return lpdf
    
    def to(self, device):
        self.c = self.c.to(device)
        return self

class Uniform:
    def __init__(self, low, high):
        super().__init__()
        self.low = torch.tensor(low)
        self.high = torch.tensor(high)
        self.name = "uniform"

    def log_pdf(self, x):
        lb = self.low.le(x).type_as(self.low)
        ub = self.high.gt(x).type_as(self.low)
        lpdf = torch.log(lb.mul(ub)) - torch.log(self.high - self.low)
        return lpdf.sum(dim=-1)
    
    def to(self, device):
        self.low = self.low.to(device)
        self.high = self.high.to(device)
        return self