r"""General purpose helpers."""

__all__ = ['bisection', 'broadcast', 'gauss_legendre']

import numpy as np
import torch
import torch.nn as nn

from functools import lru_cache
from torch import Tensor
from typing import *


def bisection(
    f: Callable[[Tensor], Tensor],
    a: Tensor,
    b: Tensor,
    n: int = 16,
) -> Tensor:
    r"""Applies the bisection method to find a root :math:`x` of a function
    :math:`f(x)` between the bounds :math:`a` an :math:`b`.

    Wikipedia:
        https://wikipedia.org/wiki/Bisection_method

    Arguments:
        f: A univariate function :math:`f(x)`.
        a: The bound :math:`a` such that :math:`f(a) \leq 0`.
        b: The bound :math:`b` such that :math:`0 \leq f(b)`.
        n: The number of iterations.

    Example:
        >>> f = torch.cos
        >>> a = torch.tensor(2.0)
        >>> b = torch.tensor(1.0)
        >>> bisection(f, a, b, n=16)
        tensor(1.5708)
    """

    with torch.no_grad():
        for _ in range(n):
            c = (a + b) / 2

            mask = f(c) < 0

            a = torch.where(mask, c, a)
            b = torch.where(mask, b, c)

    return (a + b) / 2


def broadcast(*tensors: Tensor, ignore: Union[int, List[int]] = 0) -> List[Tensor]:
    r"""Broadcasts tensors together.

    The term broadcasting describes how PyTorch treats tensors with different shapes
    during arithmetic operations. In short, if possible, dimensions that have
    different sizes are expanded (without making copies) to be compatible.

    Arguments:
        ignore: The number(s) of dimensions not to broadcast.

    Example:
        >>> x = torch.rand(3, 1, 2)
        >>> y = torch.rand(4, 5)
        >>> x, y = broadcast(x, y, ignore=1)
        >>> x.shape
        torch.Size([3, 4, 2])
        >>> y.shape
        torch.Size([3, 4, 5])
    """

    if type(ignore) is int:
        ignore = [ignore] * len(tensors)

    dims = [t.dim() - i for t, i in zip(tensors, ignore)]
    common = torch.broadcast_shapes(*(t.shape[:i] for t, i in zip(tensors, dims)))

    return [torch.broadcast_to(t, common + t.shape[i:]) for t, i in zip(tensors, dims)]


class AttachLimits(torch.autograd.Function):
    r"""Attaches the limits of integration to the computational graph."""

    @staticmethod
    def forward(
        ctx,
        f: Callable[[Tensor], Tensor],
        a: Tensor,
        b: Tensor,
        area: Tensor,
    ) -> Tensor:
        ctx.f = f
        ctx.save_for_backward(a, b)

        return area

    @staticmethod
    def backward(ctx, grad_area: Tensor) -> Tuple[Tensor, ...]:
        f = ctx.f
        a, b = ctx.saved_tensors

        if ctx.needs_input_grad[1]:
            grad_a = -f(a) * grad_area
        else:
            grad_a = None

        if ctx.needs_input_grad[2]:
            grad_b = f(b) * grad_area
        else:
            grad_b = None

        return None, grad_a, grad_b, grad_area


def gauss_legendre(
    f: Callable[[Tensor], Tensor],
    a: Tensor,
    b: Tensor,
    n: int = 3,
) -> Tensor:
    r"""Estimates the definite integral of :math:`f` from :math:`a` to :math:`b`
    using a :math:`n`-point Gauss-Legendre quadrature.

    .. math:: \int_a^b f(x) ~ dx \approx (b - a) \sum_{i = 1}^n w_i f(x_i)

    Wikipedia:
        https://wikipedia.org/wiki/Gauss-Legendre_quadrature

    Arguments:
        f: A univariate function :math:`f(x)`.
        a: The lower limit :math:`a`.
        b: The upper limit :math:`b`.
        n: The number of points :math:`n` at which the function is evaluated.

    Example:
        >>> f = lambda x: torch.exp(-x**2)
        >>> a, b = torch.tensor([-0.69, 4.2])
        >>> gauss_legendre(f, a, b, n=16)
        tensor(1.4807)
    """

    nodes, weights = leggauss(n, dtype=a.dtype, device=a.device)
    nodes = torch.lerp(
        a[..., None].detach(),
        b[..., None].detach(),
        nodes,
    ).movedim(-1, 0)

    area = (b - a).detach() * torch.tensordot(weights, f(nodes), dims=1)

    return AttachLimits.apply(f, a, b, area)


@lru_cache(maxsize=None)
def leggauss(n: int, **kwargs) -> Tuple[Tensor, Tensor]:
    r"""Returns the nodes and weights for a :math:`n`-point Gauss-Legendre
    quadrature over the interval :math:`[0, 1]`.

    See :func:`numpy.polynomial.legendre.leggauss`.

    Arguments:
        n: The number of points :math:`n`.

    Example:
        >>> nodes, weights = leggauss(3)
        >>> nodes
        tensor([0.1127, 0.5000, 0.8873])
        >>> weights
        tensor([0.2778, 0.4444, 0.2778])
    """

    nodes, weights = np.polynomial.legendre.leggauss(n)

    nodes = (nodes + 1) / 2
    weights = weights / 2

    kwargs.setdefault('dtype', torch.get_default_dtype())

    return (
        torch.as_tensor(nodes, **kwargs),
        torch.as_tensor(weights, **kwargs),
    )
