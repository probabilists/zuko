r"""Mixture models."""

__all__ = [
    "GMM",
]

import torch
import torch.nn as nn

from math import prod
from torch import Tensor
from torch.distributions import (
    Distribution,
    MultivariateNormal,
)
from typing import Sequence, Tuple

from .distributions import DiagNormal, Mixture
from .lazy import LazyDistribution
from .nn import MLP
from .utils import unpack


class GMM(LazyDistribution):
    r"""Creates a Gaussian mixture model (GMM).

    .. math:: p(X | c) = \sum_{i = 1}^K w_i(c) \, \mathcal{N}(X | \mu_i(c), \Sigma_i(c))

    Wikipedia:
        https://wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model

    Arguments:
        features: The number of features.
        context: The number of context features.
        components: The number of components :math:`K` in the mixture.
        covariance_type: The type of covariance matrix parameterization to use.
            One of :py:`['full', 'diagonal', 'spherical']`.
        tied: Whether to tie the covariance parameters across components.
        epsilon: A numerical stability term.
        kwargs: Keyword arguments passed to :class:`zuko.nn.MLP`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        components: int = 2,
        covariance_type: str = "full",
        tied: bool = False,
        epsilon: float = 1e-6,
        **kwargs,
    ):
        super().__init__()

        self.components = components
        self.covariance_type = covariance_type
        self.tied = tied
        self.epsilon = epsilon

        shapes = _get_gmm_shapes(components, features, covariance_type, tied)

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

        if self.covariance_type == "full":
            return self._forward_full(*phi)
        elif self.covariance_type == "diagonal":
            return self._forward_diagonal(*phi)
        elif self.covariance_type == "spherical":
            return self._forward_diagonal(*phi)
        else:
            raise ValueError(f"Unknown covariance type '{self.covariance_type}'.")

    def _forward_full(
        self, logits: Tensor, loc: Tensor, diag: Tensor, tril: Tensor
    ) -> Distribution:
        scale = torch.diag_embed(diag.exp() + self.epsilon)
        mask = torch.tril(torch.ones_like(scale, dtype=bool), diagonal=-1)
        scale = torch.masked_scatter(scale, mask, tril)

        return Mixture(MultivariateNormal(loc=loc, scale_tril=scale), logits)

    def _forward_diagonal(self, logits: Tensor, loc: Tensor, diag: Tensor) -> Distribution:
        scale = diag.exp() + self.epsilon

        return Mixture(DiagNormal(loc=loc, scale=scale), logits)

    @torch.no_grad()
    def initialize(self, x: Tensor, strategy: str):
        r"""Initializes the components of the model.

        Note:
            For more information on the clustering strategies see
            https://scikit-learn.org/dev/modules/mixture.html#choice-of-the-initialization-method

        Arguments:
            x: The feature samples, with shape :math:`(N, D)`.
            strategy: The clustering strategy. One of :py:`['random', 'kmeans', 'kmeans++']`.
        """

        N, _ = x.shape

        assert N > self.components, (
            f"The number of samples ({N}) should be larger than the number of components ({self.components})."
        )

        if strategy == "random":
            centers = _cluster_random(x, self.components)
        elif strategy == "kmeans":
            centers = _cluster_kmeans(x, self.components)
        elif strategy == "kmeans++":
            centers = _cluster_kmeans_pp(x, self.components)
        else:
            raise ValueError(f"Unkown clustering strategy '{strategy}'.")

        match = torch.cdist(x, centers).argmin(dim=-1)
        match = torch.nn.functional.one_hot(match, num_classes=self.components).to(dtype=x.dtype)

        probs = torch.sum(match, dim=0) / torch.sum(match)
        means = torch.sum(match[:, :, None] * x[:, None, :], dim=0) / torch.sum(
            match[:, :, None], dim=0
        )

        if self.covariance_type == "full":
            covs = _estimate_full_cov(x, match, self.tied)
        elif self.covariance_type == "diagonal":
            covs = _estimate_diagonal_cov(x, match, self.tied)
        elif self.covariance_type == "spherical":
            covs = _estimate_spherical_cov(x, match, self.tied)
        else:
            raise ValueError(f"Unkown covariance type '{self.covariance_type}'.")

        if torch.is_tensor(covs):
            params = (probs.log(), means, covs)
        else:
            params = (probs.log(), means, *covs)

        assert all(p.shape == s for p, s in zip(params, self.shapes))

        if hasattr(self, "hyper"):
            self.hyper[-1].weight.mul_(1e-2)
            self.hyper[-1].bias.copy_(torch.cat([p.flatten() for p in params]))
        else:
            for p, p_ in zip(self.phi, params):
                p.copy_(p_)


def _get_gmm_shapes(
    components: int,
    features: int,
    covariance_type: str,
    tied: bool,
) -> Sequence[int]:
    leading = 1 if tied else components

    shapes = [
        (components,),  # probabilities
        (components, features),  # mean
    ]

    if covariance_type == "full":
        shapes.extend([
            (leading, features),  # diagonal
            (leading, features * (features - 1) // 2),  # off diagonal
        ])
    elif covariance_type == "diagonal":
        shapes.extend([
            (leading, features),  # diagonal
        ])
    elif covariance_type == "spherical":
        shapes.extend([
            (leading, 1),  # diagonal
        ])
    else:
        raise ValueError(f"Unknown covariance type '{covariance_type}'.")

    return shapes


def _estimate_full_cov(x: Tensor, match: Tensor, tied: bool) -> Tuple[Tensor, Tensor]:
    _, D = x.shape
    _, K = match.shape

    covariances = []

    for k in range(K):
        covariances.append(torch.cov(x.T, aweights=match[:, k]))

    covariances = torch.stack(covariances)

    if tied:
        covariances = covariances.mean(dim=0, keepdim=True)

    lower = torch.linalg.cholesky(covariances)

    diag = torch.diagonal(lower, dim1=-2, dim2=-1)
    tril = lower[(..., *torch.tril_indices(D, D, offset=-1))]

    return diag.log(), tril


def _estimate_diagonal_cov(x: Tensor, match: Tensor, tied: bool) -> Tensor:
    _, D = x.shape
    _, K = match.shape

    diag = []

    for k in range(K):
        for d in range(D):
            diag.append(torch.cov(x[:, d], aweights=match[:, k]))

    diag = torch.stack(diag).reshape(K, D)

    if tied:
        diag = diag.mean(dim=0, keepdim=True)

    return diag.log()


def _estimate_spherical_cov(x: Tensor, match: Tensor, tied: bool) -> Tensor:
    diag = _estimate_diagonal_cov(x, match, tied)
    diag = diag.exp().mean(dim=-1, keepdim=True)

    return diag.log()


def _cluster_random(x: Tensor, components: int):
    N, _ = x.shape

    idx = torch.multinomial(torch.ones(N, device=x.device), components)
    centers = x[idx]

    return centers


def _cluster_kmeans(x: Tensor, components: int, iterations: int = 7):
    N, _ = x.shape

    centers = _cluster_kmeans_pp(x, components)

    for _ in range(iterations):
        match = torch.cdist(x, centers).argmin(dim=-1)
        match = torch.nn.functional.one_hot(match, num_classes=components).to(dtype=x.dtype)

        idx = torch.multinomial(torch.ones(N, device=x.device), components)
        centers = torch.where(
            torch.sum(match[:, :, None], dim=0) > 0,
            torch.sum(match[:, :, None] * x[:, None, :], dim=0)
            / torch.sum(match[:, :, None], dim=0),
            x[idx],
        )

    return centers


def _cluster_kmeans_pp(x: Tensor, components: int):
    N, _ = x.shape

    idx = torch.multinomial(torch.ones(N, device=x.device), components)
    centers = x[idx]

    mask = torch.zeros((N, components), dtype=torch.bool, device=x.device)
    mask[idx, 0] = True

    for k in range(1, components):
        dist = torch.cdist(x, centers[:k])
        dist[mask[:, :k]] = 0
        dist = dist.min(dim=-1).values

        idx = torch.multinomial(dist.square(), 1)
        centers[k] = x[idx]
        mask[idx, k] = True

    return centers
