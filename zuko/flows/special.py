r"""Special flows."""

__all__ = [
    'NSF',
    'NCSF',
    'SOSPF',
]

import torch

from math import pi
from torch.distributions import *
from typing import *

from .autoregressive import MAF
from .core import *
from ..distributions import *
from ..transforms import *


class NSF(MAF):
    r"""Creates a neural spline flow (NSF) with monotonic rational-quadratic spline
    transformations.

    By default, transformations are fully autoregressive. Coupling transformations can
    be obtained by setting :py:`passes=2`.

    References:
        | Neural Spline Flows (Durkan et al., 2019)
        | https://arxiv.org/abs/1906.04032

    Arguments:
        features: The number of features.
        context: The number of context features.
        bins: The number of bins :math:`K`.
        kwargs: Keyword arguments passed to :class:`zuko.flows.autoregressive.MAF`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        bins: int = 8,
        **kwargs,
    ):
        super().__init__(
            features=features,
            context=context,
            univariate=MonotonicRQSTransform,
            shapes=[(bins,), (bins,), (bins - 1,)],
            **kwargs,
        )


class NCSF(NSF):
    r"""Creates a neural circular spline flow (NCSF).

    Note:
        Features are assumed to lie in the half-open interval :math:`[-\pi, \pi[`.

    References:
        | Normalizing Flows on Tori and Spheres (Rezende et al., 2020)
        | https://arxiv.org/abs/2002.02428

    Arguments:
        features: The number of features.
        context: The number of context features.
        kwargs: Keyword arguments passed to :class:`NSF`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        **kwargs,
    ):
        super().__init__(features, context, **kwargs)

        for t in self.transforms:
            t.univariate = self.circular_spline

        self.base = Unconditional(
            BoxUniform,
            torch.full((features,), -pi - 1e-5),
            torch.full((features,), pi + 1e-5),
            buffer=True,
        )

    @staticmethod
    def circular_spline(*args) -> Transform:
        return ComposedTransform(
            CircularShiftTransform(bound=pi),
            MonotonicRQSTransform(*args, bound=pi),
        )


class SOSPF(MAF):
    r"""Creates a sum-of-squares polynomial flow (SOSPF).

    References:
        | Sum-of-Squares Polynomial Flow (Jaini et al., 2019)
        | https://arxiv.org/abs/1905.02325

    Arguments:
        features: The number of features.
        context: The number of context features.
        degree: The degree :math:`L` of polynomials.
        polynomials: The number of polynomials :math:`K`.
        kwargs: Keyword arguments passed to :class:`zuko.flows.autoregressive.MAF`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        degree: int = 3,
        polynomials: int = 2,
        **kwargs,
    ):
        super().__init__(
            features=features,
            context=context,
            univariate=SOSPolynomialTransform,
            shapes=[(polynomials, degree + 1), ()],
            **kwargs,
        )
