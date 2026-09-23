r"""Planar flows."""

__all__ = [
    "PNF",
    "PlanarLazyTransform",
]

import torch
import torch.nn as nn

from math import prod
from torch import Tensor
from torch.distributions import Transform

from ..distributions import DiagNormal
from ..lazy import Flow, LazyTransform, UnconditionalDistribution
from ..nn import MLP
from ..transforms import PlanarTransform
from ..utils import unpack


class PlanarLazyTransform(LazyTransform):
    r"""Creates a lazy planar transformation.

    See also:
        :class:`zuko.transforms.PlanarTransform`

    References:
        | Variational Inference with Normalizing Flows (Rezende et al., 2015)
        | https://arxiv.org/abs/1505.05770

    Arguments:
        features: The number of features.
        context: The number of context features.
        kwargs: Keyword arguments passed to :class:`zuko.nn.MLP`.

    Example:
        >>> t = PlanarLazyTransform(3, 4)
        >>> t
        PlanarLazyTransform(
          (hyper): MLP(
            (0): Linear(in_features=4, out_features=64, bias=True)
            (1): ReLU()
            (2): Linear(in_features=64, out_features=64, bias=True)
            (3): ReLU()
            (4): Linear(in_features=64, out_features=7, bias=True)
          )
        )
        >>> x = torch.randn(3)
        >>> x
        tensor([-1.3411,  0.1149,  0.6243])
        >>> c = torch.randn(4)
        >>> y = t(c)(x)
        >>> t(c).inv(y)
        tensor([-1.3411,  0.1149,  0.6243], grad_fn=<SubBackward0>)
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        **kwargs,
    ):
        super().__init__()

        self.shapes = [(features,), (features,), ()]
        self.total = sum(prod(s) for s in self.shapes)

        if context > 0:
            self.hyper = MLP(context, self.total, **kwargs)
        else:
            self.phi = nn.ParameterList(torch.randn(s) for s in self.shapes)

    def forward(self, c: Tensor = None) -> Transform:
        if c is None:
            phi = self.phi
        else:
            phi = unpack(self.hyper(c), self.shapes)

        return PlanarTransform(*phi)


class PNF(Flow):
    r"""Creates a planar normalizing flow (PNF).

    Note:
        Evaluating densities is cheap, but sampling requires inverting planar
        transformations with the bisection method. When sampling is the bottleneck, as
        in variational inference, the flow can be reversed with
        :py:`Flow(flow.transform.inv, flow.base)`, as in the original paper.

    See also:
        :class:`PlanarLazyTransform`

    References:
        | Variational Inference with Normalizing Flows (Rezende et al., 2015)
        | https://arxiv.org/abs/1505.05770

    Arguments:
        features: The number of features.
        context: The number of context features.
        transforms: The number of planar transformations.
        kwargs: Keyword arguments passed to :class:`PlanarLazyTransform`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        transforms: int = 3,
        **kwargs,
    ):
        transforms = [
            PlanarLazyTransform(
                features=features,
                context=context,
                **kwargs,
            )
            for _ in range(transforms)
        ]

        base = UnconditionalDistribution(
            DiagNormal,
            loc=torch.zeros(features),
            scale=torch.ones(features),
            buffer=True,
        )

        super().__init__(transforms, base)
