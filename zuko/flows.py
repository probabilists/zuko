r"""Parameterized flows and transformations."""

__all__ = [
    'DistributionModule',
    'TransformModule',
    'FlowModule',
    'GMM',
    'ElementWiseTransform',
    'MaskedAutoregressiveTransform',
    'MAF',
    'NSF',
    'NCSF',
    'SOSPF',
    'NeuralAutoregressiveTransform',
    'NAF',
    'UnconstrainedNeuralAutoregressiveTransform',
    'UNAF',
    'GeneralCouplingTransform',
    'GCF',
    'FFJTransform',
    'CNF',
]

import abc
import torch
import torch.nn as nn

from functools import partial
from math import ceil, pi, prod
from textwrap import indent
from torch import Tensor, BoolTensor, LongTensor, Size
from torch.distributions import *
from typing import *

from .distributions import *
from .transforms import *
from .nn import *
from .utils import broadcast, unpack


class DistributionModule(nn.Module, abc.ABC):
    r"""Abstract distribution module."""

    @abc.abstractmethod
    def forward(c: Tensor = None) -> Distribution:
        r"""
        Arguments:
            c: A context :math:`c`.

        Returns:
            A distribution :math:`p(X | c)`.
        """

        pass


class TransformModule(nn.Module, abc.ABC):
    r"""Abstract transformation module."""

    @abc.abstractmethod
    def forward(c: Tensor = None) -> Transform:
        r"""
        Arguments:
            c: A context :math:`c`.

        Returns:
            A transformation :math:`y = f(x | c)`.
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
        transforms: Sequence[TransformModule],
        base: DistributionModule,
    ):
        super().__init__()

        self.transforms = nn.ModuleList(transforms)
        self.base = base

    def forward(self, c: Tensor = None) -> NormalizingFlow:
        r"""
        Arguments:
            c: A context :math:`c`.

        Returns:
            A normalizing flow :math:`p(X | c)`.
        """

        transform = ComposedTransform(*(t(c) for t in self.transforms))

        if c is None:
            base = self.base(c)
        else:
            base = self.base(c).expand(c.shape[:-1])

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

    def forward(self, c: Tensor = None) -> Any:
        return self.meta(
            *self._parameters.values(),
            *self._buffers.values(),
            **self.kwargs,
        )


class Parameters(nn.ParameterList):
    r"""Creates a list of parameters."""

    def extra_repr(self) -> str:
        lines = [
            f'({i}): Tensor(shape={tuple(p.shape)})'
            for i, p in enumerate(self)
        ]

        return indent('\n'.join(lines), '  ')


class GMM(DistributionModule):
    r"""Creates a Gaussian mixture model (GMM).

    .. math:: p(X | c) = \sum_{i = 1}^K w_i(c) \, \mathcal{N}(X | \mu_i(c), \Sigma_i(c))

    Arguments:
        features: The number of features.
        context: The number of context features.
        components: The number of components :math:`K` in the mixture.
        kwargs: Keyword arguments passed to :class:`zuko.nn.MLP`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        components: int = 2,
        **kwargs,
    ):
        super().__init__()

        shapes = [
            (components,),  # probabilities
            (components, features),  # mean
            (components, features),  # diagonal
            (components, features * (features - 1) // 2),  # off diagonal
        ]

        self.shapes = shapes
        self.total = sum(prod(s) for s in shapes)

        if context > 0:
            self.hyper = MLP(context, self.total, **kwargs)
        else:
            self.phi = Parameters(torch.randn(*s) for s in shapes)

    def forward(self, c: Tensor = None) -> Distribution:
        if c is None:
            phi = self.phi
        else:
            phi = self.hyper(c)
            phi = unpack(phi, self.shapes)

        logits, loc, diag, tril = phi

        scale = torch.diag_embed(diag.exp() + 1e-5)
        mask = torch.tril(torch.ones_like(scale, dtype=bool), diagonal=-1)
        scale = torch.masked_scatter(scale, mask, tril)

        return Mixture(MultivariateNormal(loc=loc, scale_tril=scale), logits)


class ElementWiseTransform(TransformModule):
    r"""Creates an element-wise transformation.

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
            self.phi = Parameters(torch.randn(features, *s) for s in shapes)

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
        >>> c = torch.randn(4)
        >>> y = t(c)(x)
        >>> t(c).inv(y)
        tensor([-0.9485,  1.5290,  0.2018])
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        passes: int = None,
        order: LongTensor = None,
        univariate: Callable[..., Transform] = MonotonicAffineTransform,
        shapes: Sequence[Size] = ((), ()),
        **kwargs,
    ):
        super().__init__()

        # Univariate transformation
        self.univariate = univariate
        self.shapes = shapes
        self.total = sum(prod(s) for s in shapes)

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
        out_order = torch.repeat_interleave(self.order, self.total)
        adjacency = out_order[:, None] > in_order

        # Hyper network
        self.hyper = MaskedMLP(adjacency, **kwargs)

    def extra_repr(self) -> str:
        base = self.univariate(*map(torch.randn, self.shapes))
        order = self.order.tolist()

        if len(order) > 10:
            order = order[:5] + [...] + order[-5:]
            order = str(order).replace('Ellipsis', '...')

        return '\n'.join([
            f'(base): {base}',
            f'(order): {order}',
        ])

    def meta(self, c: Tensor, x: Tensor) -> Transform:
        if c is not None:
            x = torch.cat(broadcast(x, c, ignore=1), dim=-1)

        phi = self.hyper(x)
        phi = phi.unflatten(-1, (-1, self.total))
        phi = unpack(phi, self.shapes)

        return DependentTransform(self.univariate(*phi), 1)

    def forward(self, c: Tensor = None) -> Transform:
        return AutoregressiveTransform(partial(self.meta, c), self.passes)


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
        >>> c = torch.randn(4)
        >>> x = flow(c).sample()
        >>> x
        tensor([-1.7154, -0.4401,  0.7505])
        >>> flow(c).log_prob(x)
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
            self.transforms.insert(i, Unconditional(SoftclipTransform, bound=6.0))


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
        >>> c = torch.randn(4)
        >>> y = t(c)(x)
        >>> t(c).inv(y)
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


class NAF(FlowModule):
    r"""Creates a neural autoregressive flow (NAF).

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
        unconstrained: Whether to use unconstrained or regular monotonic networks.
        kwargs: Keyword arguments passed to :class:`NeuralAutoregressiveTransform`.
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
            NeuralAutoregressiveTransform(
                features=features,
                context=context,
                order=torch.randperm(features) if randperm else orders[i % 2],
                **kwargs,
            )
            for i in range(transforms)
        ]

        for i in reversed(range(len(transforms))):
            transforms.insert(i, Unconditional(SoftclipTransform, bound=6.0))

        base = Unconditional(
            DiagNormal,
            torch.zeros(features),
            torch.ones(features),
            buffer=True,
        )

        super().__init__(transforms, base)


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
        >>> c = torch.randn(4)
        >>> y = t(c)(x)
        >>> t(c).inv(y)
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


class UNAF(FlowModule):
    r"""Creates an unconstrained neural autoregressive flow (UNAF).

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
        kwargs: Keyword arguments passed to :class:`UnconstrainedNeuralAutoregressiveTransform`.
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
            UnconstrainedNeuralAutoregressiveTransform(
                features=features,
                context=context,
                order=torch.randperm(features) if randperm else orders[i % 2],
                **kwargs,
            )
            for i in range(transforms)
        ]

        for i in reversed(range(len(transforms))):
            transforms.insert(i, Unconditional(SoftclipTransform, bound=6.0))

        base = Unconditional(
            DiagNormal,
            torch.zeros(features),
            torch.ones(features),
            buffer=True,
        )

        super().__init__(transforms, base)


class GeneralCouplingTransform(TransformModule):
    r"""Creates a general coupling transformation.

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
          (order): [0, 1, 0]
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
        tensor([-0.8743,  0.6232,  1.2439])
        >>> c = torch.randn(4)
        >>> y = t(c)(x)
        >>> t(c).inv(y)
        tensor([-0.8743,  0.6232,  1.2439])
    """

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
        self.register_buffer('mask', None)

        if mask is None:
            self.mask = torch.arange(features) % 2 == 1
        else:
            self.mask = mask

        features_a = self.mask.sum().item()
        features_b = features - features_a

        # Hyper network
        self.hyper = MLP(features_a + context, features_b * self.total, **kwargs)

    def extra_repr(self) -> str:
        base = self.univariate(*map(torch.randn, self.shapes))
        mask = self.mask.int().tolist()

        if len(mask) > 10:
            mask = mask[:5] + [...] + mask[-5:]
            mask = str(mask).replace('Ellipsis', '...')

        return '\n'.join([
            f'(base): {base}',
            f'(mask): {mask}',
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


class GCF(FlowModule):
    r"""Creates a general coupling flow (GCF).

    Arguments:
        features: The number of features.
        context: The number of context features.
        transforms: The number of coupling transformations.
        randmask: Whether random coupling masks are used or not. If :py:`False`,
            use alternating checkered masks.
        features are in ascending (descending) order for even
            (odd) transformations.
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

        base = Unconditional(
            DiagNormal,
            torch.zeros(features),
            torch.ones(features),
            buffer=True,
        )

        super().__init__(temp, base)


class FFJTransform(TransformModule):
    r"""Creates a free-form Jacobian (FFJ) transformation.

    References:
        | FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models (Grathwohl et al., 2018)
        | https://arxiv.org/abs/1810.01367

    Arguments:
        features: The number of features.
        context: The number of context features.
        freqs: The number of time embedding frequencies.
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
        tensor([ 0.1777,  1.0139, -1.0370])
        >>> c = torch.randn(4)
        >>> y = t(c)(x)
        >>> t(c).inv(y)
        tensor([ 0.1777,  1.0139, -1.0370])
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        freqs: int = 3,
        exact: bool = True,
        **kwargs,
    ):
        super().__init__()

        kwargs.setdefault('activation', nn.ELU)

        self.ode = MLP(features + context + 2 * freqs, features, **kwargs)

        self.register_buffer('times', torch.tensor((0.0, 1.0)))
        self.register_buffer('freqs', torch.arange(1, freqs + 1) * pi)

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
            exact=self.exact,
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
        kwargs: Keyword arguments passed to :class:`FFJTransform`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        **kwargs,
    ):
        transforms = [
            FFJTransform(
                features=features,
                context=context,
                **kwargs,
            )
        ]

        base = Unconditional(
            DiagNormal,
            torch.zeros(features),
            torch.ones(features),
            buffer=True,
        )

        super().__init__(transforms, base)
