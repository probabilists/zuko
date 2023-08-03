r"""Mixture models."""

__all__ = [
    'GMM',
]

import torch
import torch.nn as nn

from math import prod
from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal
from typing import *

from .core import *
from ..distributions import Mixture
from ..nn import MLP
from ..utils import unpack


class GMM(LazyDistribution):
    r"""Creates a Gaussian mixture model (GMM).

    .. math:: p(X | c) = \sum_{i = 1}^K w_i(c) \, \mathcal{N}(X | \mu_i(c), \Sigma_i(c))

    Wikipedia:
        https://wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model

    Arguments:
        features: The number of features.
        context: The number of context features.
        components: The number of components :math:`K` in the mixture.
        kwargs: Keyword arguments passed to :class:`zuko.nn.MLP`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        components: int = 2,
        **kwargs,
    ):
        super().__init__()

        shapes = [
            (components,),  # probabilities
            (components, features),  # mean
            (components, features),  # diagonal
            (components, features * (features - 1) // 2),  # off diagonal
        ]

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

        logits, loc, diag, tril = phi

        scale = torch.diag_embed(diag.exp() + 1e-5)
        mask = torch.tril(torch.ones_like(scale, dtype=bool), diagonal=-1)
        scale = torch.masked_scatter(scale, mask, tril)

        return Mixture(MultivariateNormal(loc=loc, scale_tril=scale), logits)
