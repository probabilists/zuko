r"""Neural flows and transformations."""

__all__ = [
    "MNN",
    "NAF",
    "UMNN",
    "UNAF",
]

import torch
import torch.nn as nn

from functools import partial
from torch import Tensor
from torch.distributions import Transform
from typing import Any, Dict

from .autoregressive import MaskedAutoregressiveTransform
from ..distributions import DiagNormal
from ..lazy import Flow, UnconditionalDistribution, UnconditionalTransform
from ..nn import MLP, MonotonicMLP
from ..transforms import (
    MonotonicTransform,
    SoftclipTransform,
    UnconstrainedMonotonicTransform,
)
from ..utils import broadcast


class MNN(nn.Module):
    r"""Creates a monotonic neural network (MNN).

    The monotonic neural network is parametrized by its internal positive weights, which
    are independent of the features and context. To modulate its behavior, it receives
    as input a signal vector that can depend on the features and context.

    See also:
        :class:`zuko.transforms.MonotonicTransform`

    References:
        | Neural Autoregressive Flows (Huang et al., 2018)
        | https://arxiv.org/abs/1804.00779

    Arguments:
        signal: The number of signal features of the monotonic network.
        kwargs: Keyword arguments passed to :class:`zuko.nn.MonotonicMLP`.
    """

    def __init__(self, signal: int = 16, **kwargs):
        super().__init__()

        self.network = MonotonicMLP(1 + signal, 1, **kwargs)

    def f(self, signal: Tensor, x: Tensor) -> Tensor:
        y = self.network(torch.cat(broadcast(x[..., None], signal, ignore=1), dim=-1))
        y = y.squeeze(dim=-1)

        return y

    def forward(self, signal: Tensor) -> Transform:
        return MonotonicTransform(
            f=partial(self.f, signal),
            phi=(signal, *self.parameters()),
        )


class UMNN(nn.Module):
    r"""Creates an unconstrained monotonic neural network (UMNN).

    The integrand neural network is parametrized by its internal weights, which are
    independent of the features and context. To modulate its behavior, it receives as
    input a signal vector that can depend on the features and context.

    See also:
        :class:`zuko.transforms.UnconstrainedMonotonicTransform`

    References:
        | Unconstrained Monotonic Neural Networks (Wehenkel et al., 2019)
        | https://arxiv.org/abs/1908.05164

    Arguments:
        signal: The number of signal features of the integrand network.
        kwargs: Keyword arguments passed to :class:`zuko.nn.MLP`.
    """

    def __init__(self, signal: int = 16, **kwargs):
        super().__init__()

        kwargs.setdefault("activation", nn.ELU)

        self.integrand = MLP(1 + signal, 1, **kwargs)

    def g(self, signal: Tensor, x: Tensor) -> Tensor:
        dx = self.integrand(torch.cat(broadcast(x[..., None], signal, ignore=1), dim=-1))
        dx = dx.squeeze(dim=-1)

        return torch.exp(dx / (1 + abs(dx / 9)))  # in [1e-4, 1e4]

    def forward(self, signal: Tensor, constant: Tensor) -> Transform:
        return UnconstrainedMonotonicTransform(
            g=partial(self.g, signal),
            C=constant,
            phi=(signal, *self.parameters()),
        )


class NAF(Flow):
    r"""Creates a neural autoregressive flow (NAF).

    Warning:
        Invertibility is only guaranteed for features within the interval :math:`[-10,
        10]`. It is recommended to standardize features (zero mean, unit variance)
        before training.

    References:
        | Neural Autoregressive Flows (Huang et al., 2018)
        | https://arxiv.org/abs/1804.00779

    Arguments:
        features: The number of features.
        context: The number of context features.
        transforms: The number of autoregressive transformations.
        randperm: Whether features are randomly permuted between transformations or not.
            If :py:`False`, features are in ascending (descending) order for even
            (odd) transformations.
        signal: The number of signal features of the monotonic network.
        network: Keyword arguments passed to :class:`MNN`.
        kwargs: Keyword arguments passed to :class:`zuko.flows.autoregressive.MaskedAutoregressiveTransform`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        transforms: int = 3,
        randperm: bool = False,
        signal: int = 16,
        network: Dict[str, Any] = {},  # noqa: B006
        **kwargs,
    ):
        orders = [
            torch.arange(features),
            torch.flipud(torch.arange(features)),
        ]

        transforms = [
            MaskedAutoregressiveTransform(
                features=features,
                context=context,
                order=torch.randperm(features) if randperm else orders[i % 2],
                univariate=MNN(signal=signal, stack=features, **network),
                shapes=[(signal,)],
                **kwargs,
            )
            for i in range(transforms)
        ]

        for i in reversed(range(1, len(transforms))):
            transforms.insert(i, UnconditionalTransform(SoftclipTransform, bound=11.0))

        base = UnconditionalDistribution(
            DiagNormal,
            torch.zeros(features),
            torch.ones(features),
            buffer=True,
        )

        super().__init__(transforms, base)


class UNAF(Flow):
    r"""Creates an unconstrained neural autoregressive flow (UNAF).

    Warning:
        Invertibility is only guaranteed for features within the interval :math:`[-10,
        10]`. It is recommended to standardize features (zero mean, unit variance)
        before training.

    References:
        | Unconstrained Monotonic Neural Networks (Wehenkel et al., 2019)
        | https://arxiv.org/abs/1908.05164

    Arguments:
        features: The number of features.
        context: The number of context features.
        transforms: The number of autoregressive transformations.
        randperm: Whether features are randomly permuted between transformations or not.
            If :py:`False`, features are in ascending (descending) order for even
            (odd) transformations.
        signal: The number of signal features of the monotonic network.
        network: Keyword arguments passed to :class:`UMNN`.
        kwargs: Keyword arguments passed to :class:`zuko.flows.autoregressive.MaskedAutoregressiveTransform`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        transforms: int = 3,
        randperm: bool = False,
        signal: int = 16,
        network: Dict[str, Any] = {},  # noqa: B006
        **kwargs,
    ):
        orders = [
            torch.arange(features),
            torch.flipud(torch.arange(features)),
        ]

        transforms = [
            MaskedAutoregressiveTransform(
                features=features,
                context=context,
                order=torch.randperm(features) if randperm else orders[i % 2],
                univariate=UMNN(signal=signal, stack=features, **network),
                shapes=[(signal,), ()],
                **kwargs,
            )
            for i in range(transforms)
        ]

        for i in reversed(range(1, len(transforms))):
            transforms.insert(i, UnconditionalTransform(SoftclipTransform, bound=11.0))

        base = UnconditionalDistribution(
            DiagNormal,
            torch.zeros(features),
            torch.ones(features),
            buffer=True,
        )

        super().__init__(transforms, base)
