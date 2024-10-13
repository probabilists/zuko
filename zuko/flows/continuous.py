r"""Continuous flows and transformations."""

__all__ = [
    "CNF",
    "FFJTransform",
]

import torch
import torch.nn as nn

from functools import partial
from math import pi
from torch import Tensor
from torch.distributions import Transform

from ..distributions import DiagNormal
from ..lazy import Flow, LazyTransform, UnconditionalDistribution
from ..nn import MLP
from ..transforms import FreeFormJacobianTransform
from ..utils import broadcast


class FFJTransform(LazyTransform):
    r"""Creates a lazy free-form Jacobian (FFJ) transformation.

    See also:
        :class:`zuko.transforms.FreeFormJacobianTransform`

    References:
        | FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models (Grathwohl et al., 2018)
        | https://arxiv.org/abs/1810.01367

    Arguments:
        features: The number of features.
        context: The number of context features.
        freqs: The number of time embedding frequencies.
        atol: The absolute integration tolerance.
        rtol: The relative integration tolerance.
        exact: Whether the exact log-determinant of the Jacobian or an unbiased
            stochastic estimate thereof is calculated.
        kwargs: Keyword arguments passed to :class:`zuko.nn.MLP`.

    Example:
        >>> t = FFJTransform(3, 4)
        >>> t
        FFJTransform(
          (ode): MLP(
            (0): Linear(in_features=13, out_features=64, bias=True)
            (1): ELU(alpha=1.0)
            (2): Linear(in_features=64, out_features=64, bias=True)
            (3): ELU(alpha=1.0)
            (4): Linear(in_features=64, out_features=3, bias=True)
          )
        )
        >>> x = torch.randn(3)
        >>> x
        tensor([ 0.6365, -0.3181,  1.1519])
        >>> c = torch.randn(4)
        >>> y = t(c)(x)
        >>> t(c).inv(y)
        tensor([ 0.6364, -0.3181,  1.1519],
               grad_fn=<AdaptiveCheckpointAdjointBackward>)
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        freqs: int = 3,
        atol: float = 1e-6,
        rtol: float = 1e-5,
        exact: bool = True,
        **kwargs,
    ):
        super().__init__()

        kwargs.setdefault("activation", nn.ELU)

        self.ode = MLP(features + context + 2 * freqs, features, **kwargs)

        self.register_buffer("times", torch.tensor((0.0, 1.0)))
        self.register_buffer("freqs", torch.arange(1, freqs + 1) * pi)

        self.atol = atol
        self.rtol = rtol
        self.exact = exact

    def f(self, t: Tensor, x: Tensor, c: Tensor = None) -> Tensor:
        t = self.freqs * t[..., None]
        t = torch.cat((t.cos(), t.sin()), dim=-1)

        if c is None:
            x = torch.cat(broadcast(t, x, ignore=1), dim=-1)
        else:
            x = torch.cat(broadcast(t, x, c, ignore=1), dim=-1)

        return self.ode(x)

    def forward(self, c: Tensor = None) -> Transform:
        return FreeFormJacobianTransform(
            f=partial(self.f, c=c),
            t0=self.times[0],
            t1=self.times[1],
            phi=self.parameters() if c is None else (c, *self.parameters()),
            atol=self.atol,
            rtol=self.rtol,
            exact=self.exact,
        )


class CNF(Flow):
    r"""Creates a continuous normalizing flow (CNF) with a free-form Jacobian
    transformation.

    References:
        | Neural Ordinary Differential Equations (Chen el al., 2018)
        | https://arxiv.org/abs/1806.07366

        | FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models (Grathwohl et al., 2018)
        | https://arxiv.org/abs/1810.01367

    Arguments:
        features: The number of features.
        context: The number of context features.
        kwargs: Keyword arguments passed to :class:`FFJTransform`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        **kwargs,
    ):
        transform = FFJTransform(
            features=features,
            context=context,
            **kwargs,
        )

        base = UnconditionalDistribution(
            DiagNormal,
            torch.zeros(features),
            torch.ones(features),
            buffer=True,
        )

        super().__init__(transform, base)
