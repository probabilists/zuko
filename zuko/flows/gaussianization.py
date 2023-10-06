r"""Gaussianization flows."""

__all__ = [
    'ElementWiseTransform',
    'GF',
]

import torch
import torch.nn as nn

from math import prod
from torch import Tensor, Size
from torch.distributions import Transform
from typing import *

from .core import *
from ..distributions import DiagNormal
from ..transforms import *
from ..nn import MLP
from ..utils import unpack


class ElementWiseTransform(LazyTransform):
    r"""Creates a lazy element-wise transformation.

    Arguments:
        features: The number of features.
        context: The number of context features.
        univariate: The univariate transformation constructor.
        shapes: The shapes of the univariate transformation parameters.
        kwargs: Keyword arguments passed to :class:`zuko.nn.MLP`.

    Example:
        >>> t = ElementWiseTransform(3, 4)
        >>> t
        ElementWiseTransform(
          (base): MonotonicAffineTransform()
          (hyper): MLP(
            (0): Linear(in_features=4, out_features=64, bias=True)
            (1): ReLU()
            (2): Linear(in_features=64, out_features=64, bias=True)
            (3): ReLU()
            (4): Linear(in_features=64, out_features=6, bias=True)
          )
        )
        >>> x = torch.randn(3)
        >>> x
        tensor([2.1983,  -1.3182,  0.0329])
        >>> c = torch.randn(4)
        >>> y = t(c)(x)
        >>> t(c).inv(y)
        tensor([2.1983,  -1.3182,  0.0329])
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        univariate: Callable[..., Transform] = MonotonicAffineTransform,
        shapes: Sequence[Size] = ((), ()),
        **kwargs,
    ):
        super().__init__()

        self.univariate = univariate
        self.shapes = shapes
        self.total = sum(prod(s) for s in shapes)

        if context > 0:
            self.hyper = MLP(context, features * self.total, **kwargs)
        else:
            self.phi = nn.ParameterList(torch.randn(features, *s) for s in shapes)

    def extra_repr(self) -> str:
        base = self.univariate(*map(torch.randn, self.shapes))

        return '\n'.join([
            f'(base): {base}',
        ])

    def forward(self, c: Tensor = None) -> Transform:
        if c is None:
            phi = self.phi
        else:
            phi = self.hyper(c)
            phi = phi.unflatten(-1, (-1, self.total))
            phi = unpack(phi, self.shapes)

        return DependentTransform(self.univariate(*phi), 1)


class GF(Flow):
    r"""Creates a gaussianization flow (GF).

    Warning:
        Invertibility is only guaranteed for features within the interval :math:`[-10,
        10]`. It is recommended to standardize features (zero mean, unit variance)
        before training.

    See also:
        :class:`zuko.transforms.GaussianizationTransform`

    References:
        | Gaussianization Flows (Meng et al., 2020)
        | https://arxiv.org/abs/2003.01941

    Arguments:
        features: The number of features.
        context: The number of context features.
        transforms: The number of coupling transformations.
        components: The number of mixture components in each transformation.
        kwargs: Keyword arguments passed to :class:`ElementWiseTransform`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        transforms: int = 3,
        components: int = 8,
        **kwargs,
    ):
        transforms = [
            ElementWiseTransform(
                features=features,
                context=context,
                univariate=GaussianizationTransform,
                shapes=[(components,), (components,)],
                **kwargs,
            )
            for _ in range(transforms)
        ]

        for i in reversed(range(1, len(transforms))):
            transforms.insert(
                i,
                Unconditional(
                    RotationTransform,
                    torch.randn(features, features),
                ),
            )

        base = Unconditional(
            DiagNormal,
            torch.zeros(features),
            torch.ones(features),
            buffer=True,
        )

        super().__init__(transforms, base)
