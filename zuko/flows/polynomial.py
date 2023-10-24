r"""Polynomial flows."""

__all__ = [
    'SOSPF',
]

import torch

from typing import *

from .autoregressive import MAF
from .core import *
from ..transforms import SoftclipTransform, SOSPolynomialTransform


class SOSPF(MAF):
    r"""Creates a sum-of-squares polynomial flow (SOSPF).

    Warning:
        Invertibility is only guaranteed for features within the interval :math:`[-10,
        10]`. It is recommended to standardize features (zero mean, unit variance)
        before training.

    See also:
        :class:`zuko.transforms.SOSPolynomialTransform`

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
        degree: int = 4,
        polynomials: int = 3,
        **kwargs,
    ):
        super().__init__(
            features=features,
            context=context,
            univariate=SOSPolynomialTransform,
            shapes=[(polynomials, degree + 1), ()],
            **kwargs,
        )

        transforms = self.transform.transforms

        for i in reversed(range(1, len(transforms))):
            transforms.insert(i, Unconditional(SoftclipTransform, bound=11.0))
