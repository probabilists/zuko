r"""Parameterized flows and autoregressive transformations."""

__all__ = [
    'DistributionModule',
    'TransformModule',
    'FlowModule',
    'MaskedAutoregressiveTransform',
    'MAF',
    'NSF',
    'SOSPF',
    'NeuralAutoregressiveTransform',
    'UnconstrainedNeuralAutoregressiveTransform',
    'NAF',
]

import abc
import torch
import torch.nn as nn

from functools import partial
from math import ceil
from torch import Tensor, LongTensor, Size
from typing import *

from .distributions import *
from .transforms import *
from .nn import MLP, MaskedMLP, MonotonicMLP
from .utils import broadcast


class DistributionModule(nn.Module, abc.ABC):
    r"""Abstract distribution module."""

    @abc.abstractmethod
    def forward(y: Tensor = None) -> Distribution:
        r"""
        Arguments:
            y: A context :math:`y`.

        Returns:
            A distribution :math:`p(x | y)`.
        """

        pass


class TransformModule(nn.Module, abc.ABC):
    r"""Abstract transformation module."""

    @abc.abstractmethod
    def forward(y: Tensor = None) -> Transform:
        r"""
        Arguments:
            y: A context :math:`y`.

        Returns:
            A transformation :math:`z = f(x | y)`.
        """

        pass


class FlowModule(DistributionModule):
    r"""Creates a normalizing flow module.

    Arguments:
        transforms: A list of transformation modules.
        base: A distribution module.
    """

    def __init__(
        self,
        transforms: List[TransformModule],
        base: DistributionModule,
    ):
        super().__init__()

        self.transforms = nn.ModuleList(transforms)
        self.base = base

    def forward(self, y: Tensor = None) -> NormalizingFlow:
        r"""
        Arguments:
            y: A context :math:`y`.

        Returns:
            A normalizing flow :math:`p(x | y)`.
        """

        transforms = [t(y) for t in self.transforms]

        if y is None:
            base = self.base(y)
        else:
            base = self.base(y).expand(y.shape[:-1])

        return NormalizingFlow(transforms, base)


class Unconditional(nn.Module):
    r"""Creates a module that registers the positional arguments of a function.
    The function is evaluated during the forward pass and the result is returned.

    Arguments:
        meta: An arbitrary function.
        args: The positional tensor arguments passed to `meta`.
        buffer: Whether tensors are registered as buffer or parameter.
        kwargs: The keyword arguments passed to `meta`.
    """

    def __init__(
        self,
        meta: Callable[..., Any],
        *args: Tensor,
        buffer: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.meta = meta

        for i, arg in enumerate(args):
            if buffer:
                self.register_buffer(f'_{i}', arg)
            else:
                self.register_parameter(f'_{i}', nn.Parameter(arg))

        self.kwargs = kwargs

    def __repr__(self) -> str:
        return repr(self.forward())

    def forward(self, y: Tensor = None) -> Any:
        return self.meta(
            *self._parameters.values(),
            *self._buffers.values(),
            **self.kwargs,
        )


class MaskedAutoregressiveTransform(TransformModule):
    r"""Creates a masked autoregressive transformation.

    References:
        | Masked Autoregressive Flow for Density Estimation (Papamakarios et al., 2017)
        | https://arxiv.org/abs/1705.07057

    Arguments:
        features: The number of features.
        context: The number of context features.
        passes: The number of sequential passes for the inverse transformation. If
            :py:`None`, use the number of features instead, making the transformation
            fully autoregressive. Coupling corresponds to :py:`passes=2`.
        order: The feature ordering. If :py:`None`, use :py:`range(features)` instead.
        univariate: The univariate transformation constructor.
        shapes: The shapes of the univariate transformation parameters.
        kwargs: Keyword arguments passed to :class:`zuko.nn.MaskedMLP`.

    Example:
        >>> t = MaskedAutoregressiveTransform(3, 4)
        >>> t
        MaskedAutoregressiveTransform(
          (base): MonotonicAffineTransform()
          (order): [0, 1, 2]
          (hyper): MaskedMLP(
            (0): MaskedLinear(in_features=7, out_features=64, bias=True)
            (1): ReLU()
            (2): MaskedLinear(in_features=64, out_features=64, bias=True)
            (3): ReLU()
            (4): MaskedLinear(in_features=64, out_features=6, bias=True)
          )
        )
        >>> x = torch.randn(3)
        >>> x
        tensor([-0.9485,  1.5290,  0.2018])
        >>> y = torch.randn(4)
        >>> z = t(y)(x)
        >>> t(y).inv(z)
        tensor([-0.9485,  1.5290,  0.2018])
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        passes: int = None,
        order: LongTensor = None,
        univariate: Callable[..., Transform] = MonotonicAffineTransform,
        shapes: List[Size] = [(), ()],
        **kwargs,
    ):
        super().__init__()

        # Univariate transformation
        self.univariate = univariate
        self.shapes = list(map(Size, shapes))
        self.sizes = [s.numel() for s in self.shapes]

        # Adjacency
        self.register_buffer('order', None)

        if passes is None:
            passes = features

        if order is None:
            order = torch.arange(features)

        self.passes = min(max(passes, 1), features)
        self.order = torch.div(order, ceil(features / self.passes), rounding_mode='floor')

        in_order = torch.cat((self.order, torch.full((context,), -1)))
        out_order = torch.repeat_interleave(self.order, sum(self.sizes))
        adjacency = out_order[:, None] > in_order

        # Hyper network
        self.hyper = MaskedMLP(adjacency, **kwargs)

    def extra_repr(self) -> str:
        base = self.univariate(*map(torch.randn, self.shapes))
        order = self.order.tolist()

        if len(order) > 11:
            order = str(order[:5] + [...] + order[-5:]).replace('Ellipsis', '...')

        return '\n'.join([
            f'(base): {base}',
            f'(order): {order}',
        ])

    def meta(self, y: Tensor, x: Tensor) -> Transform:
        if y is not None:
            x = torch.cat(broadcast(x, y, ignore=1), dim=-1)

        params = self.hyper(x)
        params = params.reshape(*params.shape[:-1], -1, sum(self.sizes))

        args = params.split(self.sizes, dim=-1)
        args = [a.reshape(a.shape[:-1] + s) for a, s in zip(args, self.shapes)]

        return self.univariate(*args)

    def forward(self, y: Tensor = None) -> AutoregressiveTransform:
        return AutoregressiveTransform(partial(self.meta, y), self.passes)


class MAF(FlowModule):
    r"""Creates a masked autoregressive flow (MAF).

    References:
        | Masked Autoregressive Flow for Density Estimation (Papamakarios et al., 2017)
        | https://arxiv.org/abs/1705.07057

    Arguments:
        features: The number of features.
        context: The number of context features.
        transforms: The number of autoregressive transformations.
        randperm: Whether features are randomly permuted between transformations or not.
            If :py:`False`, features are in ascending (descending) order for even
            (odd) transformations.
        kwargs: Keyword arguments passed to :class:`MaskedAutoregressiveTransform`.

    Example:
        >>> flow = MAF(3, 4, transforms=3)
        >>> flow
        MAF(
          (transforms): ModuleList(
            (0): MaskedAutoregressiveTransform(
              (base): MonotonicAffineTransform()
              (order): [0, 1, 2]
              (hyper): MaskedMLP(
                (0): MaskedLinear(in_features=7, out_features=64, bias=True)
                (1): ReLU()
                (2): MaskedLinear(in_features=64, out_features=64, bias=True)
                (3): ReLU()
                (4): MaskedLinear(in_features=64, out_features=6, bias=True)
              )
            )
            (1): MaskedAutoregressiveTransform(
              (base): MonotonicAffineTransform()
              (order): [2, 1, 0]
              (hyper): MaskedMLP(
                (0): MaskedLinear(in_features=7, out_features=64, bias=True)
                (1): ReLU()
                (2): MaskedLinear(in_features=64, out_features=64, bias=True)
                (3): ReLU()
                (4): MaskedLinear(in_features=64, out_features=6, bias=True)
              )
            )
            (2): MaskedAutoregressiveTransform(
              (base): MonotonicAffineTransform()
              (order): [0, 1, 2]
              (hyper): MaskedMLP(
                (0): MaskedLinear(in_features=7, out_features=64, bias=True)
                (1): ReLU()
                (2): MaskedLinear(in_features=64, out_features=64, bias=True)
                (3): ReLU()
                (4): MaskedLinear(in_features=64, out_features=6, bias=True)
              )
            )
          )
          (base): DiagNormal(loc: torch.Size([3]), scale: torch.Size([3]))
        )
        >>> y = torch.randn(4)
        >>> x = flow(y).sample()
        >>> x
        tensor([-1.7154, -0.4401,  0.7505])
        >>> flow(y).log_prob(x)
        tensor(-4.4630, grad_fn=<AddBackward0>)
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        transforms: int = 3,
        randperm: bool = False,
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
                **kwargs,
            )
            for i in range(transforms)
        ]

        base = Unconditional(
            DiagNormal,
            torch.zeros(features),
            torch.ones(features),
            buffer=True,
        )

        super().__init__(transforms, base)


class NSF(MAF):
    r"""Creates a neural spline flow (NSF) with monotonic rational-quadratic spline
    transformations.

    Note:
        By default, transformations are fully autoregressive. Coupling transformations
        can be obtained by setting :py:`passes=2`.

    References:
        | Neural Spline Flows (Durkan et al., 2019)
        | https://arxiv.org/abs/1906.04032

    Arguments:
        features: The number of features.
        context: The number of context features.
        bins: The number of bins :math:`K`.
        kwargs: Keyword arguments passed to :class:`MAF`.
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
        kwargs: Keyword arguments passed to :class:`MAF`.
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

        for i in reversed(range(len(self.transforms))):
            self.transforms.insert(i, Unconditional(SoftclipTransform))


class NeuralAutoregressiveTransform(MaskedAutoregressiveTransform):
    r"""Creates a neural autoregressive transformation.

    The monotonic neural network is parametrized by its internal positive weights,
    which are independent of the features and context. To modulate its behavior, it
    receives as input a signal that is autoregressively dependent on the features
    and context.

    References:
        | Neural Autoregressive Flows (Huang et al., 2018)
        | https://arxiv.org/abs/1804.00779

    Arguments:
        features: The number of features.
        context: The number of context features.
        signal: The number of signal features of the monotonic network.
        network: Keyword arguments passed to :class:`zuko.nn.MonotonicMLP`.
        kwargs: Keyword arguments passed to :class:`MaskedAutoregressiveTransform`.

    Example:
        >>> t = NeuralAutoregressiveTransform(3, 4)
        >>> t
        NeuralAutoregressiveTransform(
          (base): MonotonicTransform()
          (order): [0, 1, 2]
          (hyper): MaskedMLP(
            (0): MaskedLinear(in_features=7, out_features=64, bias=True)
            (1): ReLU()
            (2): MaskedLinear(in_features=64, out_features=64, bias=True)
            (3): ReLU()
            (4): MaskedLinear(in_features=64, out_features=24, bias=True)
          )
          (network): MonotonicMLP(
            (0): MonotonicLinear(in_features=9, out_features=64, bias=True)
            (1): TwoWayELU(alpha=1.0)
            (2): MonotonicLinear(in_features=64, out_features=64, bias=True)
            (3): TwoWayELU(alpha=1.0)
            (4): MonotonicLinear(in_features=64, out_features=1, bias=True)
          )
        )
        >>> x = torch.randn(3)
        >>> x
        tensor([-2.3267,  1.4581, -1.6776])
        >>> y = torch.randn(4)
        >>> z = t(y)(x)
        >>> t(y).inv(z)
        tensor([-2.3267,  1.4581, -1.6776])
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        signal: int = 8,
        network: Dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__(
            features=features,
            context=context,
            univariate=self.univariate,
            shapes=[(signal,)],
            **kwargs,
        )

        self.network = MonotonicMLP(1 + signal, 1, **network)

    def f(self, signal: Tensor, x: Tensor) -> Tensor:
        return self.network(
            torch.cat(broadcast(x[..., None], signal, ignore=1), dim=-1)
        ).squeeze(dim=-1)

    def univariate(self, signal: Tensor) -> Transform:
        return MonotonicTransform(partial(self.f, signal))


class UnconstrainedNeuralAutoregressiveTransform(MaskedAutoregressiveTransform):
    r"""Creates an unconstrained neural autoregressive transformation.

    The integrand neural network is parametrized by its internal weights, which are
    independent of the features and context. To modulate its behavior, it receives as
    input a signal that is autoregressively dependent on the features and context. The
    integration constant has the same dependencies as the signal.

    References:
        | Unconstrained Monotonic Neural Networks (Wehenkel et al., 2019)
        | https://arxiv.org/abs/1908.05164

    Arguments:
        features: The number of features.
        context: The number of context features.
        signal: The number of signal features of the integrand network.
        network: Keyword arguments passed to :class:`zuko.nn.MLP`.
        kwargs: Keyword arguments passed to :class:`MaskedAutoregressiveTransform`.

    Example:
        >>> t = UnconstrainedNeuralAutoregressiveTransform(3, 4)
        >>> t
        UnconstrainedNeuralAutoregressiveTransform(
          (base): UnconstrainedMonotonicTransform()
          (order): [0, 1, 2]
          (hyper): MaskedMLP(
            (0): MaskedLinear(in_features=7, out_features=64, bias=True)
            (1): ReLU()
            (2): MaskedLinear(in_features=64, out_features=64, bias=True)
            (3): ReLU()
            (4): MaskedLinear(in_features=64, out_features=27, bias=True)
          )
          (integrand): MLP(
            (0): Linear(in_features=9, out_features=64, bias=True)
            (1): ELU(alpha=1.0)
            (2): Linear(in_features=64, out_features=64, bias=True)
            (3): ELU(alpha=1.0)
            (4): Linear(in_features=64, out_features=1, bias=True)
            (5): Softplus(beta=1, threshold=20)
          )
        )
        >>> x = torch.randn(3)
        >>> x
        tensor([-0.0103, -1.0871, -0.0667])
        >>> y = torch.randn(4)
        >>> z = t(y)(x)
        >>> t(y).inv(z)
        tensor([-0.0103, -1.0871, -0.0667])
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        signal: int = 8,
        network: Dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__(
            features=features,
            context=context,
            univariate=self.univariate,
            shapes=[(signal,), ()],
            **kwargs,
        )

        network.setdefault('activation', nn.ELU)

        self.integrand = MLP(1 + signal, 1, **network)
        self.integrand.add_module(str(len(self.integrand)), nn.Softplus())

    def g(self, signal: Tensor, x: Tensor) -> Tensor:
        return self.integrand(
            torch.cat(broadcast(x[..., None], signal, ignore=1), dim=-1)
        ).squeeze(dim=-1)

    def univariate(self, signal: Tensor, constant: Tensor) -> Transform:
        return UnconstrainedMonotonicTransform(partial(self.g, signal), constant)


class NAF(FlowModule):
    r"""Creates a neural autoregressive flow (NAF).

    References:
        | Neural Autoregressive Flows (Huang et al., 2018)
        | https://arxiv.org/abs/1804.00779

        | Unconstrained Monotonic Neural Networks (Wehenkel et al., 2019)
        | https://arxiv.org/abs/1908.05164

    Arguments:
        features: The number of features.
        context: The number of context features.
        transforms: The number of autoregressive transformations.
        randperm: Whether features are randomly permuted between transformations or not.
            If :py:`False`, features are in ascending (descending) order for even
            (odd) transformations.
        unconstrained: Whether to use unconstrained or regular monotonic networks.
        kwargs: Keyword arguments passed to :class:`NeuralAutoregressiveTransform` or
            :class:`UnconstrainedNeuralAutoregressiveTransform`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        transforms: int = 3,
        randperm: bool = False,
        unconstrained: bool = False,
        **kwargs,
    ):
        if unconstrained:
            build = UnconstrainedNeuralAutoregressiveTransform
        else:
            build = NeuralAutoregressiveTransform

        orders = [
            torch.arange(features),
            torch.flipud(torch.arange(features)),
        ]

        transforms = [
            build(
                features=features,
                context=context,
                order=torch.randperm(features) if randperm else orders[i % 2],
                **kwargs,
            )
            for i in range(transforms)
        ]

        for i in reversed(range(len(transforms))):
            transforms.insert(i, Unconditional(SoftclipTransform))

        base = Unconditional(
            DiagNormal,
            torch.zeros(features),
            torch.ones(features),
            buffer=True,
        )

        super().__init__(transforms, base)
