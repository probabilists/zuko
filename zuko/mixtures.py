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
    Independent,
    MultivariateNormal,
    Normal,
)

from .distributions import Mixture
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
        covariance_type: String describing the type of covariance parameters to use. Must be one of:

            - ‘full’: each component has its own full rank covariance matrix.

            - ‘diag’: each component has its own diagonal covariance matrix.

            - ‘spherical’: each component has its own single variance.

        tied: Whether to use tied covariance matrices. Tied covariances share the same parameters across components.
        kwargs: Keyword arguments passed to :class:`zuko.nn.MLP`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        components: int = 2,
        covariance_type: str = "full",
        tied: bool = False,
        reg_covar: float = 1e-2,
        **kwargs,
    ):
        super().__init__()

        self.feature = features
        self.context = context

        self.components = components
        self.covariance_type = covariance_type
        self.tied = tied
        self.reg_covar = reg_covar
        self.weights_init = None
        self.means_init = None
        self.covariances_init = None

        shapes = _determine_shapes(components, features, covariance_type, tied)

        self.shapes = shapes
        self.total = sum(prod(s) for s in shapes)

        if context > 0:
            self.hyper = MLP(context, self.total, **kwargs)
        else:
            self.phi = nn.ParameterList(torch.randn(*s) for s in shapes)

    def initialize(
        self,
        f: Tensor = None,
        strategy: str = "random",
        init_weights: Tensor = None,
        init_means: Tensor = None,
        init_covariances: Tensor = None,
    ) -> None:
        r"""Initializes the parameters of the model. For more information on the initilisation strategies, see: https://scikit-learn.org/dev/modules/mixture.html#choice-of-the-initialization-method

        Arguments:
            f: Feature samples. Should be of shape `(n_samples, n_features)`, and have at least as many samples as
                components.
            strategy: The initialization strategy. Must be one of:

                - 'random': Random initialization.

                - 'kmeans': K-means initialization.

                - 'kmeans++': K-means++ initialization.
        """
        n_samples = f.shape[0]

        if n_samples < self.components:
            raise ValueError(
                f"Number of samples ({f.shape[0]}) must be greater than number of components ({self.components})."
            )

        self.init_strategy = strategy

        # resp is a matrix assigning each sample to a component
        # the assignment is soft, i.e. the sum of each row is 1

        if self.init_strategy == "random":
            resp = _initialize_random(f, self.components)

        elif self.init_strategy == "kmeans++":
            resp = _initialize_kmeans_plus_plus(f, self.components)

        elif self.init_strategy == "kmeans":
            resp = _initialize_kmeans(f, self.components)

        else:
            raise ValueError(f"Invalid initialization strategy: {self.init_strategy}")

        self._initialize_weights(f, resp, init_weights, init_means, init_covariances)

    def forward(self, c: Tensor = None) -> Distribution:
        if c is None:
            phi = self.phi
        else:
            phi = self.hyper(c)
            phi = unpack(phi, self.shapes)

        forward_methods = {
            "full": self._forward_full,
            "diag": self._forward_diag_or_spherical,
            "spherical": self._forward_diag_or_spherical,
        }

        if self.covariance_type not in forward_methods:
            raise ValueError(f"Invalid covariance type: {self.covariance_type}")

        return forward_methods[self.covariance_type](phi)

    def _forward_full(self, phi):
        logits, loc, diag, tril = phi

        scale = torch.diag_embed(diag.exp() + self.reg_covar)
        mask = torch.tril(torch.ones_like(scale, dtype=bool), diagonal=-1)
        scale = torch.masked_scatter(scale, mask, tril)
        return Mixture(MultivariateNormal(loc=loc, scale_tril=scale), logits)

    def _forward_diag_or_spherical(self, phi):
        logits, loc, diag = phi

        diag = diag.exp() + self.reg_covar

        return Mixture(Independent(Normal(loc, diag), 1), logits)

    def _initialize_weights(self, f, resp, weights_init, means_init, covariances_init):
        weights, means, diag, off_diag = None, None, None, None

        if resp is not None:
            weights, means, diag, off_diag = self._estimate_gaussian_parameters(
                f,
                resp,
            )

        weights_ = weights if weights_init is None else weights_init
        means_ = means if means_init is None else means_init

        if covariances_init is None:
            diag_ = diag
            off_diag_ = off_diag
        else:
            diag_ = torch.diagonal(self.covariances_init, dim1=-2, dim2=-1)
            off_diag_ = self.covariances_init[
                ..., torch.tril_indices(self.covariances_init.shape[-1], -1)
            ]
        diag_ = torch.log(diag_ - self.reg_covar)

        param_list = [
            weights_,
            means_,
            diag_,
        ]
        if off_diag_ is not None:
            param_list.append(off_diag_)

        if self.context > 0:
            # If the context variables are not normalised, this can lead to quite slow inital training convergence
            with torch.no_grad():
                self.hyper[-1].weight.mul_(1e-2)
                self.hyper[-1].bias.copy_(torch.cat([p.flatten() for p in param_list]))
        else:
            for p, param in zip(self.phi, param_list):
                p.data = param

    def _estimate_gaussian_parameters(self, f, resp):
        nk = resp.sum(axis=0) + 10 * torch.finfo(resp.dtype).eps
        means = ((resp[..., :, None] * f[..., None, :]) / nk[:, None]).sum(0)

        diag, off_diag = {
            "full": _estimate_gaussian_covariances_full,
            "diag": _estimate_gaussian_covariances_diag,
            "spherical": _estimate_gaussian_covariances_spherical,
        }[self.covariance_type](resp, f, nk, means, self.tied)
        return nk / f.shape[0], means, diag, off_diag


def _determine_shapes(components, features, covariance_type, tied):
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
    elif covariance_type == "diag":
        shapes.extend([
            (leading, features),  # diagonal
        ])
    elif covariance_type == "spherical":
        shapes.extend([
            (leading, 1),  # diagonal
        ])
    else:
        raise ValueError(
            f"Invalid covariance type: {covariance_type} (choose from full, diag, or spherical)"
        )
    return shapes


def _estimate_gaussian_covariances_full(resp, f, nk, means, tied):
    n_components, n_features = means.shape
    covariances = torch.empty((n_components, n_features, n_features))
    for k in range(n_components):
        covariances[k] = torch.cov((f[resp[:, k] > 0]).T, aweights=resp[resp[:, k] > 0, k])

    if tied:
        covariances = covariances.mean(dim=0)

    covariances = torch.linalg.cholesky(covariances)

    diag = torch.diagonal(covariances, dim1=-2, dim2=-1)
    tril = covariances[
        ...,
        torch.tril_indices(n_features, n_features, -1)[0],
        torch.tril_indices(n_features, n_features, -1)[1],
    ]
    return diag, tril


def _estimate_gaussian_covariances_diag(resp, f, nk, means, tied):
    n_components, n_features = means.shape

    diag = torch.empty((n_components, n_features))
    for k in range(n_components):
        diag[k] = torch.cov((f[resp[:, k] > 0]).T, aweights=resp[resp[:, k] > 0, k]).diagonal()

    if tied:
        diag = diag.mean(dim=0)

    return diag, None


def _estimate_gaussian_covariances_spherical(resp, f, nk, means, tied):
    diag, _ = _estimate_gaussian_covariances_diag(resp, f, nk, means, tied)

    return diag.mean(dim=-1).unsqueeze(-1), None


def _initialize_random(f, components):
    n_samples = f.shape[0]
    resp = torch.rand((n_samples, components))
    resp /= resp.sum(dim=1)[:, None]
    return resp


def _initialize_kmeans(f, components):
    n_samples = f.shape[0]
    centers = torch.empty((components, f.shape[1]))
    idx = torch.randint(0, n_samples, (components,))
    centers = f[idx]

    for _ in range(10):  # number of iterations
        dist = torch.cdist(f, centers).pow(2)
        resp = torch.zeros((n_samples, components))
        resp[torch.arange(n_samples), dist.argmin(dim=-1)] = 1

        for k in range(components):
            if resp[:, k].sum() == 0:
                centers[k] = f[torch.randint(0, n_samples, (1,))]
            else:
                centers[k] = (resp[:, k][:, None] * f).sum(dim=0) / resp[:, k].sum()

    dist = torch.cdist(f, centers).pow(2)
    resp = torch.zeros((n_samples, components))
    resp[torch.arange(n_samples), dist.argmin(dim=-1)] = 1
    return resp


def _initialize_kmeans_plus_plus(f, components):
    n_samples = f.shape[0]
    centers = torch.empty((components, f.shape[1]))
    idx = torch.randint(0, n_samples, (components,))
    centers = f[idx]

    mask = torch.zeros((n_samples, components), dtype=torch.bool, device=f.device)
    mask[idx.view(-1), 0] = True

    for k_i in range(1, components):
        dist = torch.cdist(f, centers[:k_i]).pow(2)
        dist[mask[:, :k_i]] = 0
        dist = dist.min(dim=-1).values
        idx = torch.multinomial(dist, 1)
        centers[k_i] = f[idx]
        mask[idx, k_i] = True

    resp = torch.zeros((n_samples, components))
    dist = torch.cdist(f, centers).pow(2)
    resp[torch.arange(n_samples), dist.argmin(dim=-1)] = 1
    return resp
