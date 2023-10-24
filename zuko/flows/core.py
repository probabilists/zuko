r"""Core building blocks."""

from __future__ import annotations

__all__ = [
    'LazyDistribution',
    'LazyTransform',
    'LazyComposedTransform',
    'Flow',
    'Unconditional',
]

import abc
import torch
import torch.nn as nn

from torch import Tensor
from torch.distributions import Distribution, Transform
from typing import *

from ..distributions import NormalizingFlow
from ..transforms import ComposedTransform


class LazyDistribution(nn.Module, abc.ABC):
    r"""Abstract lazy distribution.

    A lazy distribution is a module that builds and returns a distribution
    :math:`p(X | c)` within its forward pass, given a context :math:`c`.

    See also:
        :class:`torch.distributions.distribution.Distribution`
    """

    @abc.abstractmethod
    def forward(self, c: Any = None) -> Distribution:
        r"""
        Arguments:
            c: A context :math:`c`.

        Returns:
            A distribution :math:`p(X | c)`.
        """

        pass


class LazyTransform(nn.Module, abc.ABC):
    r"""Abstract lazy transformation.

    A lazy transformation is a module that builds and returns a transformation
    :math:`y = f(x | c)` within its forward pass, given a context :math:`c`.

    See also:
        :class:`torch.distributions.transforms.Transform`
    """

    @abc.abstractmethod
    def forward(self, c: Any = None) -> Transform:
        r"""
        Arguments:
            c: A context :math:`c`.

        Returns:
            A transformation :math:`y = f(x | c)`.
        """

        pass

    @property
    def inv(self) -> LazyTransform:
        r"""A lazy inverse transformation :math:`x = f^{-1}(y | c)`."""

        return LazyInverse(self)


class LazyInverse(LazyTransform):
    r"""Creates a lazy inverse transformation.

    Arguments:
        transform: A lazy transformation :math:`y = f(x | c)`.
    """

    def __init__(self, transform: LazyTransform):
        super().__init__()

        self.transform = transform

    def forward(self, c: Any = None) -> Transform:
        return self.transform(c).inv

    @property
    def inv(self) -> LazyTransform:
        return self.transform


class LazyComposedTransform(LazyTransform):
    r"""Creates a lazy composed transformation.

    See also:
        :class:`zuko.transforms.ComposedTransform`

    Arguments:
        transforms: A sequence of lazy transformations :math:`f_i`.
    """

    def __init__(self, *transforms: LazyTransform):
        super().__init__()

        self.transforms = nn.ModuleList(transforms)

    def __repr__(self) -> str:
        return repr(self.transforms).replace('ModuleList', 'LazyComposedTransform', 1)

    def forward(self, c: Any = None) -> Transform:
        r"""
        Arguments:
            c: A context :math:`c`.

        Returns:
            A transformation :math:`y = f_n \circ \dots \circ f_0(x | c)`.
        """

        return ComposedTransform(*(t(c) for t in self.transforms))


class Flow(LazyDistribution):
    r"""Creates a lazy normalizing flow.

    See also:
        :class:`zuko.distributions.NormalizingFlow`

    Arguments:
        transform: A lazy transformation or sequence of lazy transformations.
        base: A lazy distribution.
    """

    def __init__(
        self,
        transform: Union[LazyTransform, Sequence[LazyTransform]],
        base: LazyDistribution,
    ):
        super().__init__()

        if isinstance(transform, LazyTransform):
            self.transform = transform
        else:
            self.transform = LazyComposedTransform(*transform)

        self.base = base

    def forward(self, c: Tensor = None) -> NormalizingFlow:
        r"""
        Arguments:
            c: A context :math:`c`.

        Returns:
            A normalizing flow :math:`p(X | c)`.
        """

        transform = self.transform(c)

        if c is None:
            base = self.base(c)
        else:
            base = self.base(c).expand(c.shape[:-1])

        return NormalizingFlow(transform, base)


class Unconditional(nn.Module):
    r"""Creates an unconditional lazy module from a constructor.

    Typically, the constructor returns a distribution or transformation. The positional
    arguments of the constructor are registered as buffers or parameters.

    Arguments:
        meta: An arbitrary constructor function.
        args: The positional tensor arguments passed to `meta`.
        buffer: Whether tensors are registered as buffers or parameters.
        kwargs: The keyword arguments passed to `meta`.

    Examples:
        >>> mu, sigma = torch.zeros(3), torch.ones(3)
        >>> d = Unconditional(DiagNormal, mu, sigma, buffer=True)
        >>> d()
        DiagNormal(loc: torch.Size([3]), scale: torch.Size([3]))
        >>> d().sample()
        tensor([-0.6687, -0.9690,  1.7461])

        >>> t = Unconditional(ExpTransform)
        >>> t()
        ExpTransform()
        >>> x = torch.randn(3)
        >>> t()(x)
        tensor([0.5523, 0.7997, 0.9189])
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

    def extra_repr(self) -> str:
        if isinstance(self.meta, nn.Module):
            return ''
        else:
            return repr(self.forward())

    def forward(self, c: Tensor = None) -> Any:
        r"""
        Arguments:
            c: A context :math:`c`. This argument is always ignored.

        Returns:
            :py:`meta(*args, **kwargs)`
        """

        return self.meta(
            *self._parameters.values(),
            *self._buffers.values(),
            **self.kwargs,
        )
