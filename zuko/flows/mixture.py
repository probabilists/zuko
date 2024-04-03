r"""Mixture models."""

__all__ = [
    'GMM',
]

import torch
import torch.nn as nn


# isort: local
from .core import LazyDistribution
from ..distributions import Mixture
from ..nn import MLP
from ..utils import unpack
from math import prod
from torch import Tensor
from torch.distributions import (
    Distribution,
    Independent,
    LowRankMultivariateNormal,
    MultivariateNormal,
    Normal,
)


class GMM(LazyDistribution):
    r"""Creates a Gaussian mixture model (GMM).

    .. math:: p(X | c) = \sum_{i = 1}^K w_i(c) \, \mathcal{N}(X | \mu_i(c), \Sigma_i(c))

    Wikipedia:
        https://wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model

    Arguments:
        features: The number of features.
        context: The number of context features.
        components: The number of components :math:`K` in the mixture.
        covariance_type: String describing the type of covariance parameters to use. Must be one of:

            - ‘full’: each component has its own general covariance matrix.

            - ‘tied’: all components share the same general covariance matrix.

            - ‘diag’: each component has its own diagonal covariance matrix.

            - ‘spherical’: each component has its own single variance.

            - 'lowrank': each component has its own low-rank covariance matrix.

        cov_rank: The rank of the low-rank covariance matrix. Only used when `covariance_type` is 'lowrank'.
        kwargs: Keyword arguments passed to :class:`zuko.nn.MLP`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        components: int = 2,
        covariance_type: str = 'full',
        cov_rank: int = None,
        **kwargs,
    ):
        super().__init__()

        shapes = [
            (components,),  # probabilities
            (components, features),  # mean
        ]
        if covariance_type == 'full':
            shapes.extend([
                (components, features),  # diagonal
                (components, features * (features - 1) // 2),  # off diagonal
            ])
        elif covariance_type == 'tied':
            shapes.extend([
                (1, features),  # diagonal
                (1, features * (features - 1) // 2),  # off diagonal
            ])
        elif covariance_type == 'lowrank':
            if cov_rank is None:
                raise ValueError('cov_rank must be specified when covariance_type is lowrank')
            shapes.extend([
                (components, features),  # diagonal
                (components, features * cov_rank),  # low-rank
            ])
        elif covariance_type == 'diag':
            shapes.extend([
                (components, features),  # diagonal
            ])
        elif covariance_type == 'spherical':
            shapes.extend([
                (components, 1),  # diagonal
            ])
        else:
            raise ValueError(
                f'Invalid covariance type: {covariance_type} (choose from full, tied, lowrank, diag, or spherical)'
            )

        self.covariance_type = covariance_type
        self.cov_rank = cov_rank
        self.shapes = shapes
        self.total = sum(prod(s) for s in shapes)

        if context > 0:
            self.hyper = MLP(context, self.total, **kwargs)
        else:
            self.phi = nn.ParameterList(torch.randn(*s) for s in shapes)

    def forward(self, c: Tensor = None) -> Distribution:
        if c is None:
            phi = self.phi
        else:
            phi = self.hyper(c)
            phi = unpack(phi, self.shapes)

        if self.covariance_type in ['full', 'tied']:
            logits, loc, diag, tril = phi
            scale = torch.diag_embed(diag.exp() + 1e-5)
            mask = torch.tril(torch.ones_like(scale, dtype=bool), diagonal=-1)
            scale = torch.masked_scatter(scale, mask, tril)
            # expanded automatically for tied covariance
            return Mixture(MultivariateNormal(loc=loc, scale_tril=scale), logits)

        if self.covariance_type == 'lowrank':
            logits, loc, diag, lowrank = phi
            diag = diag.exp() + 1e-5
            lowrank = lowrank.reshape(lowrank.shape[0], lowrank.shape[1], self.cov_rank)
            return Mixture(
                LowRankMultivariateNormal(loc=loc, cov_factor=lowrank, cov_diag=diag), logits
            )

        elif self.covariance_type in ['diag', 'spherical']:
            logits, loc, diag = phi
            diag = diag.exp() + 1e-5
            # expanded automatically for spherical covariance
            return Mixture(Independent(Normal(loc, diag), 1), logits)
