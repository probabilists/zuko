r"""Core building blocks."""

__all__ = [
    'DistributionFactory',
    'TransformFactory',
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


class DistributionFactory(nn.Module, abc.ABC):
    r"""Abstract distribution factory.

    A distribution factory is a module that builds and returns a distribution
    :math:`p(X | c)` within its forward pass, given a context :math:`c`.
    """

    @abc.abstractmethod
    def forward(self, c: Tensor = None) -> Distribution:
        r"""
        Arguments:
            c: A context :math:`c`.

        Returns:
            A distribution :math:`p(X | c)`.
        """

        pass


class TransformFactory(nn.Module, abc.ABC):
    r"""Abstract transformation factory.

    A transformation factory is a module that builds and returns a transformation
    :math:`y = f(x | c)` within its forward pass, given a context :math:`c`.
    """

    @abc.abstractmethod
    def forward(self, c: Tensor = None) -> Transform:
        r"""
        Arguments:
            c: A context :math:`c`.

        Returns:
            A transformation :math:`y = f(x | c)`.
        """

        pass


class Flow(DistributionFactory):
    r"""Creates a normalizing flow factory.

    Arguments:
        transforms: A list of transformation factories.
        base: A distribution factory.
    """

    def __init__(
        self,
        transforms: Sequence[TransformFactory],
        base: DistributionFactory,
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
    r"""Creates an unconditional factory from a recipe.

    Typically, the recipe returns a distribution or transformation. The positional
    arguments of the recipe are registered as buffers or parameters.

    Arguments:
        recipe: An arbitrary function.
        args: The positional tensor arguments passed to `recipe`.
        buffer: Whether tensors are registered as buffers or parameters.
        kwargs: The keyword arguments passed to `recipe`.

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
        recipe: Callable[..., Any],
        *args: Tensor,
        buffer: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.recipe = recipe

        for i, arg in enumerate(args):
            if buffer:
                self.register_buffer(f'_{i}', arg)
            else:
                self.register_parameter(f'_{i}', nn.Parameter(arg))

        self.kwargs = kwargs

    def __repr__(self) -> str:
        return repr(self.forward())

    def forward(self, c: Tensor = None) -> Any:
        r"""
        Arguments:
            c: A context :math:`c`. This argument is always ignored.

        Returns:
            :py:`recipe(*args, **kwargs)`
        """

        return self.recipe(
            *self._parameters.values(),
            *self._buffers.values(),
            **self.kwargs,
        )
