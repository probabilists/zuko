r"""Spline flows."""

__all__ = [
    "NCSF",
    "NSF",
]

import torch

from functools import partial
from math import pi
from torch import Tensor
from torch.distributions import Transform

from .autoregressive import MAF
from ..distributions import BoxUniform
from ..lazy import UnconditionalDistribution
from ..transforms import CircularShiftTransform, ComposedTransform, MonotonicRQSTransform


class NSF(MAF):
    r"""Creates a neural spline flow (NSF) with monotonic rational-quadratic spline
    transformations.

    By default, transformations are fully autoregressive. Coupling transformations
    can be obtained by setting :py:`passes=2`.

    Warning:
        Spline transformations are defined over the domain :math:`[-5, 5]`. Any feature
        outside of this domain is not transformed. It is recommended to standardize
        features (zero mean, unit variance) before training.

    See also:
        :class:`zuko.transforms.MonotonicRQSTransform`

    References:
        | Neural Spline Flows (Durkan et al., 2019)
        | https://arxiv.org/abs/1906.04032

    Arguments:
        features: The number of features.
        context: The number of context features.
        bins: The number of bins :math:`K`.
        slope: The minimum slope of the spline transformation(s).
        kwargs: Keyword arguments passed to :class:`zuko.flows.autoregressive.MAF`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        bins: int = 8,
        slope: float = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__(
            features=features,
            context=context,
            univariate=partial(MonotonicRQSTransform, slope=slope),
            shapes=[(bins,), (bins,), (bins - 1,)],
            **kwargs,
        )


def CircularRQSTransform(*phi: Tensor, slope: float = 1e-3) -> Transform:
    r"""Creates a circular rational-quadratic spline (RQS) transformation."""

    return ComposedTransform(
        CircularShiftTransform(bound=pi),
        MonotonicRQSTransform(*phi, bound=pi, slope=slope),
    )


class NCSF(MAF):
    r"""Creates a neural circular spline flow (NCSF).

    Circular spline transformations are obtained by composing circular domain shifts
    with regular spline transformations. Features are assumed to lie in the half-open
    interval :math:`[-\pi, \pi[`.

    See also:
        :class:`zuko.transforms.CircularShiftTransform`

    References:
        | Normalizing Flows on Tori and Spheres (Rezende et al., 2020)
        | https://arxiv.org/abs/2002.02428

    Arguments:
        features: The number of features.
        context: The number of context features.
        bins: The number of bins :math:`K`.
        slope: The minimum slope of the spline transformation(s).
        kwargs: Keyword arguments passed to :class:`zuko.flows.autoregressive.MAF`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        bins: int = 8,
        slope: float = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__(
            features=features,
            context=context,
            univariate=partial(CircularRQSTransform, slope=slope),
            shapes=[(bins,), (bins,), (bins - 1,)],
            **kwargs,
        )

        self.base = UnconditionalDistribution(
            BoxUniform,
            lower=torch.full((features,), -pi - 1e-5),
            upper=torch.full((features,), pi + 1e-5),
            buffer=True,
        )
