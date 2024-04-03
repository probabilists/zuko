r"""Mixture models.

Warning:
    This sub-module is deprecated and will be removed in the future. Use
    :mod:`zuko.mixtures` instead.
"""

__all__ = [
    "GMM",
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


def _determine_shapes(components, features, covariance_type, tied, cov_rank):
    leading = 1 if tied else components

    shapes = [
        (components,),  # probabilities
        (components, features),  # mean
    ]
    if covariance_type == 'full':
        shapes.extend([
            (leading, features),  # diagonal
            (leading, features * (features - 1) // 2),  # off diagonal
        ])
    elif covariance_type == 'lowrank':
        if cov_rank is None:
            raise ValueError('cov_rank must be specified when covariance_type is lowrank')
        shapes.extend([
            (leading, features),  # diagonal
            (leading, features * cov_rank),  # low-rank
        ])
    elif covariance_type == 'diag':
        shapes.extend([
            (leading, features),  # diagonal
        ])
    elif covariance_type == 'spherical':
        shapes.extend([
            (leading, 1),  # diagonal
        ])
    else:
        raise ValueError(
            f'Invalid covariance type: {covariance_type} (choose from full, lowrank, diag, or spherical)'
        )
    return shapes


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

            - ‘full’: each component has its own full rank covariance matrix.

            - ’lowrank’: each component has its own low-rank covariance matrix.

            - ‘diag’: each component has its own diagonal covariance matrix.

            - ‘spherical’: each component has its own single variance.

        tied: Whether to use tied covariance matrices. Tied covariances share the same parameters across components.
        cov_rank: The rank of the low-rank covariance matrix. Only used when `covariance_type` is 'lowrank'.
        kwargs: Keyword arguments passed to :class:`zuko.nn.MLP`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        components: int = 2,
        covariance_type: str = 'full',
        tied: bool = False,
        cov_rank: int = None,
        **kwargs,
    ):
        super().__init__()

        shapes = _determine_shapes(components, features, covariance_type, tied, cov_rank)

        self.covariance_type = covariance_type
        self.tied = tied
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

        if self.covariance_type == 'full':
            logits, loc, diag, tril = phi
            scale = torch.diag_embed(diag.exp() + 1e-5)
            mask = torch.tril(torch.ones_like(scale, dtype=bool), diagonal=-1)
            scale = torch.masked_scatter(scale, mask, tril)
            # expanded automatically for tied covariances
            return Mixture(MultivariateNormal(loc=loc, scale_tril=scale), logits)

        if self.covariance_type == 'lowrank':
            logits, loc, diag, lowrank = phi
            diag = diag.exp() + 1e-5
            lowrank = lowrank.reshape(lowrank.shape[0], lowrank.shape[1], self.cov_rank)
            # expanded automatically for tied covariances
            return Mixture(
                LowRankMultivariateNormal(loc=loc, cov_factor=lowrank, cov_diag=diag), logits
            )

        elif self.covariance_type in ['diag', 'spherical']:
            logits, loc, diag = phi
            diag = diag.exp() + 1e-5
            # expanded automatically for spherical and tied covariance
            return Mixture(Independent(Normal(loc, diag), 1), logits)
