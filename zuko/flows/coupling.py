r"""Coupling flows and transformations."""

__all__ = [
    "NICE",
    "GeneralCouplingTransform",
]

import torch

from functools import partial
from math import prod
from torch import BoolTensor, Size, Tensor
from torch.distributions import Transform
from typing import Callable, Sequence

from .gaussianization import ElementWiseTransform
from ..distributions import DiagNormal
from ..lazy import Flow, LazyTransform, UnconditionalDistribution
from ..nn import MLP
from ..transforms import CouplingTransform, DependentTransform, MonotonicAffineTransform
from ..utils import broadcast, unpack


class GeneralCouplingTransform(LazyTransform):
    r"""Creates a lazy general coupling transformation.

    See also:
        :class:`zuko.transforms.CouplingTransform`

    References:
        | NICE: Non-linear Independent Components Estimation (Dinh et al., 2014)
        | https://arxiv.org/abs/1410.8516

    Arguments:
        features: The number of features.
        context: The number of context features.
        mask: The coupling mask. If :py:`None`, use a checkered mask.
        univariate: The univariate transformation constructor.
        shapes: The shapes of the univariate transformation parameters.
        kwargs: Keyword arguments passed to :class:`zuko.nn.MLP`.

    Example:
        >>> t = GeneralCouplingTransform(3, 4)
        >>> t
        GeneralCouplingTransform(
          (base): MonotonicAffineTransform()
          (mask): [0, 1, 0]
          (hyper): MLP(
            (0): Linear(in_features=5, out_features=64, bias=True)
            (1): ReLU()
            (2): Linear(in_features=64, out_features=64, bias=True)
            (3): ReLU()
            (4): Linear(in_features=64, out_features=4, bias=True)
          )
        )
        >>> x = torch.randn(3)
        >>> x
        tensor([-0.7900, -0.3259, -1.3184])
        >>> c = torch.randn(4)
        >>> y = t(c)(x)
        >>> t(c).inv(y)
        tensor([-0.7900, -0.3259, -1.3184], grad_fn=<IndexPutBackward0>)
    """

    def __new__(
        cls,
        features: int = None,
        context: int = 0,
        mask: BoolTensor = None,
        *args,
        **kwargs,
    ) -> LazyTransform:
        if features is None or features > 1:
            return super().__new__(cls)
        else:
            return ElementWiseTransform(features, context, *args, **kwargs)

    def __init__(
        self,
        features: int,
        context: int = 0,
        mask: BoolTensor = None,
        univariate: Callable[..., Transform] = MonotonicAffineTransform,
        shapes: Sequence[Size] = ((), ()),
        **kwargs,
    ):
        super().__init__()

        # Univariate transformation
        self.univariate = univariate
        self.shapes = shapes
        self.total = sum(prod(s) for s in shapes)

        # Mask
        if mask is None:
            mask = torch.arange(features) % 2 == 1
        else:
            mask = torch.as_tensor(mask, dtype=bool)

        assert mask.ndim == 1, "'mask' should be a vector."
        assert mask.shape[0] == features, f"'mask' should have {features} elements."

        features_a = mask.sum().item()
        features_b = features - features_a

        assert features_a > 0
        assert features_b > 0

        self.register_buffer("mask", mask)

        # Hyper network
        self.hyper = MLP(features_a + context, features_b * self.total, **kwargs)

    def extra_repr(self) -> str:
        base = self.univariate(*map(torch.randn, self.shapes))
        mask = self.mask.int().tolist()

        if len(mask) > 10:
            mask = mask[:5] + [...] + mask[-5:]
            mask = str(mask).replace("Ellipsis", "...")

        return "\n".join([
            f"(base): {base}",
            f"(mask): {mask}",
        ])

    def meta(self, c: Tensor, x: Tensor) -> Transform:
        if c is not None:
            x = torch.cat(broadcast(x, c, ignore=1), dim=-1)

        phi = self.hyper(x)
        phi = phi.unflatten(-1, (-1, self.total))
        phi = unpack(phi, self.shapes)

        return DependentTransform(self.univariate(*phi), 1)

    def forward(self, c: Tensor = None) -> Transform:
        return CouplingTransform(partial(self.meta, c), self.mask)


class NICE(Flow):
    r"""Creates a NICE flow.

    Affine transformations are used by default, instead of the additive transformations
    used by Dinh et al. (2014) originally.

    References:
        | NICE: Non-linear Independent Components Estimation (Dinh et al., 2014)
        | https://arxiv.org/abs/1410.8516

    Arguments:
        features: The number of features.
        context: The number of context features.
        transforms: The number of coupling transformations.
        randmask: Whether random coupling masks are used or not. If :py:`False`,
            use alternating checkered masks.
        kwargs: Keyword arguments passed to :class:`GeneralCouplingTransform`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        transforms: int = 3,
        randmask: bool = False,
        **kwargs,
    ):
        temp = []

        for i in range(transforms):
            if randmask:
                mask = torch.randperm(features) % 2 == i % 2
            else:
                mask = torch.arange(features) % 2 == i % 2

            temp.append(
                GeneralCouplingTransform(
                    features=features,
                    context=context,
                    mask=mask,
                    **kwargs,
                )
            )

        base = UnconditionalDistribution(
            DiagNormal,
            torch.zeros(features),
            torch.ones(features),
            buffer=True,
        )

        super().__init__(temp, base)
