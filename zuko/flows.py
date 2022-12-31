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
    'FreeFormJacobianTransform',
    'CNF',
    'ConvCouplingTransform',
    'Glow',
]

import abc
import torch
import torch.nn as nn

from functools import partial
from math import ceil
from torch import Tensor, LongTensor, Size
from torch.distributions import *
from typing import *

from .distributions import *
from .transforms import *
from .nn import *
from .utils import broadcast


class DistributionModule(nn.Module, abc.ABC):
    r"""Abstract distribution module."""

    @abc.abstractmethod
    def forward(y: Tensor = None) -> Distribution:
        r"""
        Arguments:
            y: A context :math:`y`.

        Returns:
            A distribution :math:`p(X | y)`.
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
            A normalizing flow :math:`p(X | y)`.
        """

        transform = ComposedTransform(*(t(y) for t in self.transforms))

        if y is None:
            base = self.base(y)
        else:
            base = self.base(y).expand(y.shape[:-1])

        return NormalizingFlow(transform, base)


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
        else:
            order = torch.as_tensor(order)

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

        if len(order) > 10:
            order = str(order[:5] + [...] + order[-5:]).replace('Ellipsis', '...')

        return '\n'.join([
            f'(base): {base}',
            f'(order): {order}',
        ])

    def meta(self, y: Tensor, x: Tensor) -> Transform:
        if y is not None:
            x = torch.cat(broadcast(x, y, ignore=1), dim=-1)

        total = sum(self.sizes)

        phi = self.hyper(x)
        phi = phi.unflatten(-1, (phi.shape[-1] // total, total))
        phi = phi.split(self.sizes, -1)
        phi = (p.unflatten(-1, s + (1,)) for p, s in zip(phi, self.shapes))
        phi = (p.squeeze(-1) for p in phi)

        return self.univariate(*phi)

    def forward(self, y: Tensor = None) -> Transform:
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
        return MonotonicTransform(
            f=partial(self.f, signal),
            phi=(signal, *self.network.parameters()),
        )


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
        return UnconstrainedMonotonicTransform(
            g=partial(self.g, signal),
            C=constant,
            phi=(signal, *self.integrand.parameters()),
        )


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


class FreeFormJacobianTransform(TransformModule):
    r"""Creates a free-form Jacobian transformation.

    References:
        | FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models (Grathwohl et al., 2018)
        | https://arxiv.org/abs/1810.01367

    Arguments:
        features: The number of features.
        context: The number of context features.
        kwargs: Keyword arguments passed to :class:`zuko.nn.MLP`.

    Example:
        >>> t = FreeFormJacobianTranform(3, 4)
        >>> t
        FreeFormJacobianTranform(
          (time): 1.000
          (ode): MLP(
            (0): Linear(in_features=8, out_features=64, bias=True)
            (1): ELU(alpha=1.0)
            (2): Linear(in_features=64, out_features=64, bias=True)
            (3): ELU(alpha=1.0)
            (4): Linear(in_features=64, out_features=3, bias=True)
          )
        )
        >>> x = torch.randn(3)
        >>> x
        tensor([ 0.1777,  1.0139, -1.0370])
        >>> y = torch.randn(4)
        >>> z = t(y)(x)
        >>> t(y).inv(z)
        tensor([ 0.1777,  1.0139, -1.0370])
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        **kwargs,
    ):
        super().__init__()

        kwargs.setdefault('activation', nn.ELU)

        self.ode = MLP(features + 1 + context, features, **kwargs)
        self.log_t = nn.Parameter(torch.tensor(0.0))

    def extra_repr(self) -> str:
        return f'(time): {self.log_t.exp().item():.3f}'

    def f(self, y: Tensor, x: Tensor, t: Tensor) -> Tensor:
        if y is None:
            x = torch.cat(broadcast(x, t[..., None], ignore=1), dim=-1)
        else:
            x = torch.cat(broadcast(x, t[..., None], y, ignore=1), dim=-1)

        return self.ode(x)

    def forward(self, y: Tensor = None) -> Transform:
        return FFJTransform(
            f=partial(self.f, y),
            time=self.log_t.exp(),
            phi=(y, *self.ode.parameters()),
        )


class CNF(FlowModule):
    r"""Creates a continuous normalizing flow (CNF) with free-form Jacobian
    transformations.

    References:
        | Neural Ordinary Differential Equations (Chen el al., 2018)
        | https://arxiv.org/abs/1806.07366

        | FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models (Grathwohl et al., 2018)
        | https://arxiv.org/abs/1810.01367

    Arguments:
        features: The number of features.
        context: The number of context features.
        transforms: The number of transformations.
        kwargs: Keyword arguments passed to :class:`FreeFormJacobianTransform`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        transforms: int = 1,
        **kwargs,
    ):
        transforms = [
            FreeFormJacobianTransform(
                features=features,
                context=context,
                **kwargs,
            )
            for _ in range(transforms)
        ]

        base = Unconditional(
            DiagNormal,
            torch.zeros(features),
            torch.ones(features),
            buffer=True,
        )

        super().__init__(transforms, base)


class ConvCouplingTransform(TransformModule):
    r"""Creates a convolution coupling transformation.

    Arguments:
        channels: The number of channels.
        context: The number of context channels.
        spatial: The number of spatial dimensions.
        univariate: The univariate transformation constructor.
        shapes: The shapes of the univariate transformation parameters.
        kwargs: Keyword arguments passed to :class:`zuko.nn.FCN`.
    """

    def __init__(
        self,
        channels: int,
        context: int = 0,
        spatial: int = 2,
        univariate: Callable[..., Transform] = MonotonicAffineTransform,
        shapes: List[Size] = [(), ()],
        **kwargs,
    ):
        super().__init__()

        self.d = channels // 2
        self.dim = -(spatial + 1)

        # Univariate transformation
        self.univariate = univariate
        self.shapes = list(map(Size, shapes))
        self.sizes = [s.numel() for s in self.shapes]

        # Hyper network
        kwargs.setdefault('activation', nn.ELU)
        kwargs.setdefault('normalize', True)

        self.hyper = FCN(
            in_channels=self.d + context,
            out_channels=(channels - self.d) * sum(self.sizes),
            spatial=spatial,
            **kwargs,
        )

    def extra_repr(self) -> str:
        base = self.univariate(*map(torch.randn, self.shapes))

        return f'(base): {base}'

    def meta(self, y: Tensor, x: Tensor) -> Transform:
        if y is not None:
            x = torch.cat(broadcast(x, y, ignore=abs(self.dim)), dim=self.dim)

        total = sum(self.sizes)

        phi = self.hyper(x)
        phi = phi.unflatten(self.dim, (phi.shape[self.dim] // total, total))
        phi = phi.movedim(self.dim, -1)
        phi = phi.split(self.sizes, -1)
        phi = (p.unflatten(-1, s + (1,)) for p, s in zip(phi, self.shapes))
        phi = (p.squeeze(-1) for p in phi)

        return self.univariate(*phi)

    def forward(self, y: Tensor = None) -> Transform:
        return CouplingTransform(partial(self.meta, y), self.d, self.dim)


class Glow(DistributionModule):
    r"""Creates a Glow-like multi-scale flow.

    References:
        | Glow: Generative Flow with Invertible 1x1 Convolutions (Kingma et al., 2018)
        | https://arxiv.org/abs/1807.03039

    Arguments:
        shape: The shape of a sample.
        context: The number of context channels at each scale.
        transforms: The number of coupling transformations at each scale.
        kwargs: Keyword arguments passed to :class:`ConvCouplingTransform`.
    """

    def __init__(
        self,
        shape: Size,
        context: Union[int, List[int]] = 0,
        transforms: List[int] = [8, 8, 8],
        **kwargs,
    ):
        super().__init__()

        channels, *space = shape
        spatial = len(space)
        dim = -len(shape)
        scales = len(transforms)

        assert all(s % 2**scales == 0 for s in space), (
            f"'shape' cannot be downscaled {scales} times"
        )

        if isinstance(context, int):
            context = [context] * len(transforms)

        self.flows = nn.ModuleList()
        self.bases = nn.ModuleList()

        for i, K in enumerate(transforms):
            flow = []
            flow.append(Unconditional(PixelShuffleTransform, dim=dim))

            channels = channels * 2**spatial
            space = [s // 2 for s in space]

            for _ in range(K):
                flow.extend([
                    Unconditional(
                        PermutationTransform,
                        torch.randperm(channels),
                        dim=dim,
                        buffer=True,
                    ),
                    Unconditional(
                        LULinearTransform,
                        torch.eye(channels),
                        dim=dim,
                    ),
                    ConvCouplingTransform(
                        channels=channels,
                        context=context[i],
                        spatial=spatial,
                        **kwargs,
                    ),
                ])

            if i < len(transforms) - 1:
                drop = channels // 2
            else:
                drop = channels

            self.flows.append(nn.ModuleList(flow))
            self.bases.append(
                Unconditional(
                    DiagNormal,
                    torch.zeros(drop, *space),
                    torch.ones(drop, *space),
                    ndims=spatial + 1,
                    buffer=True,
                )
            )

            channels = channels - drop

    def forward(self, y: Iterable[Tensor] = None) -> NormalizingFlow:
        r"""
        Arguments:
            y: A sequence of contexts :math:`y_i`. There should be one context
                per scale, but a context can be :py:`None`.

        Returns:
            A multi-scale flow :math:`p(X | y)`.
        """

        if y is None:
            y = [None] * len(self.flows)

        # Transforms
        transforms = []
        context_shapes = []

        for flow, base, y_i in zip(self.flows, self.bases, y):
            for t in flow:
                transforms.append(t(y_i))

            transforms.append(DropTransform(base(y_i)))

            if y_i is not None:
                context_shapes.append(y_i.shape)

        transform = ComposedTransform(*transforms[:-1])

        # Base
        base = transforms[-1].dist
        dim = -len(base.event_shape)

        batch_shapes = (shape[:dim] for shape in context_shapes)
        batch_shape = torch.broadcast_shapes(*batch_shapes)

        base = base.expand(batch_shape)

        return NormalizingFlow(transform, base)
