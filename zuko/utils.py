r"""General purpose helpers."""

from __future__ import annotations

__all__ = ['bisection', 'broadcast', 'gauss_legendre', 'odeint']

import numpy as np
import torch
import torch.nn as nn

from functools import lru_cache
from torch import Tensor
from typing import *


def bisection(
    f: Callable[[Tensor], Tensor],
    y: Tensor,
    a: Union[float, Tensor],
    b: Union[float, Tensor],
    n: int = 16,
    phi: Iterable[Tensor] = (),
) -> Tensor:
    r"""Applies the bisection method to find :math:`x` between the bounds :math:`a`
    an :math:`b` such that :math:`f_\phi(x)` is close to :math:`y`.

    Gradients are propagated through :math:`y` and :math:`\phi` via implicit
    differentiation.

    Wikipedia:
        https://wikipedia.org/wiki/Bisection_method

    Arguments:
        f: A univariate function :math:`f_\phi`.
        y: The target :math:`y`.
        a: The bound :math:`a` such that :math:`f_\phi(a) \leq y`.
        b: The bound :math:`b` such that :math:`y \leq f_\phi(b)`.
        n: The number of iterations.
        phi: The parameters :math:`\phi` of :math:`f_\phi`.

    Example:
        >>> f = torch.cos
        >>> y = torch.tensor(0.0)
        >>> bisection(f, y, 2.0, 1.0, n=16)
        tensor(1.5708)
    """

    a = torch.as_tensor(a).to(y)
    b = torch.as_tensor(b).to(y)

    return Bisection.apply(f, y, a, b, n, *phi)


class Bisection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        f: Callable[[Tensor], Tensor],
        y: Tensor,
        a: Tensor,
        b: Tensor,
        n: int,
        *phi: Tensor,
    ) -> Tensor:
        ctx.f = f
        ctx.save_for_backward(*phi)

        for _ in range(n):
            c = (a + b) / 2

            mask = f(c) < y

            a = torch.where(mask, c, a)
            b = torch.where(mask, b, c)

        ctx.x = (a + b) / 2

        return ctx.x

    @staticmethod
    def backward(ctx, grad_x: Tensor) -> Tuple[Tensor, ...]:
        f, x = ctx.f, ctx.x
        phi = ctx.saved_tensors

        with torch.enable_grad():
            x = x.detach().requires_grad_()
            y = f(x)

        jacobian = torch.autograd.grad(y, x, torch.ones_like(y), retain_graph=True)[0]
        grad_y = grad_x / jacobian

        if phi:
            grad_phi = torch.autograd.grad(y, phi, -grad_y, retain_graph=True)
        else:
            grad_phi = ()

        return (None, grad_y, None, None, None, *grad_phi)


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


def gauss_legendre(
    f: Callable[[Tensor], Tensor],
    a: Tensor,
    b: Tensor,
    n: int = 3,
    phi: Iterable[Tensor] = (),
) -> Tensor:
    r"""Estimates the definite integral of a function :math:`f_\phi(x)` from :math:`a`
    to :math:`b` using a :math:`n`-point Gauss-Legendre quadrature.

    .. math:: \int_a^b f_\phi(x) ~ dx \approx (b - a) \sum_{i = 1}^n w_i f_\phi(x_i)

    Wikipedia:
        https://wikipedia.org/wiki/Gauss-Legendre_quadrature

    Arguments:
        f: A univariate function :math:`f_\phi`.
        a: The lower limit :math:`a`.
        b: The upper limit :math:`b`.
        n: The number of points :math:`n` at which the function is evaluated.
        phi: The parameters :math:`\phi` of :math:`f_\phi`.

    Example:
        >>> f = lambda x: torch.exp(-x**2)
        >>> a, b = torch.tensor([-0.69, 4.2])
        >>> gauss_legendre(f, a, b, n=16)
        tensor(1.4807)
    """

    return GaussLegendre.apply(f, a, b, n, *phi)


class GaussLegendre(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        f: Callable[[Tensor], Tensor],
        a: Tensor,
        b: Tensor,
        n: int,
        *phi: Tensor,
    ) -> Tensor:
        ctx.f, ctx.n = f, n
        ctx.save_for_backward(a, b, *phi)

        return GaussLegendre.quadrature(f, a, b, n)

    @staticmethod
    def backward(ctx, grad_area: Tensor) -> Tuple[Tensor, ...]:
        f, n = ctx.f, ctx.n
        a, b, *phi = ctx.saved_tensors

        if ctx.needs_input_grad[1]:
            grad_a = -f(a) * grad_area
        else:
            grad_a = None

        if ctx.needs_input_grad[2]:
            grad_b = f(b) * grad_area
        else:
            grad_b = None

        if phi:
            with torch.enable_grad():
                area = GaussLegendre.quadrature(f, a.detach(), b.detach(), n)

            grad_phi = torch.autograd.grad(area, phi, grad_area, retain_graph=True)
        else:
            grad_phi = ()

        return (None, grad_a, grad_b, None, *grad_phi)

    @staticmethod
    @lru_cache(maxsize=None)
    def nodes(n: int, **kwargs) -> Tuple[Tensor, Tensor]:
        r"""Returns the nodes and weights for a :math:`n`-point Gauss-Legendre
        quadrature over the interval :math:`[0, 1]`.

        See :func:`numpy.polynomial.legendre.leggauss`.
        """

        nodes, weights = np.polynomial.legendre.leggauss(n)

        nodes = (nodes + 1) / 2
        weights = weights / 2

        kwargs.setdefault('dtype', torch.get_default_dtype())

        return (
            torch.as_tensor(nodes, **kwargs),
            torch.as_tensor(weights, **kwargs),
        )

    @staticmethod
    def quadrature(
        f: Callable[[Tensor], Tensor],
        a: Tensor,
        b: Tensor,
        n: int,
    ) -> Tensor:
        nodes, weights = GaussLegendre.nodes(n, dtype=a.dtype, device=a.device)
        nodes = torch.lerp(
            a[..., None],
            b[..., None],
            nodes,
        ).movedim(-1, 0)

        return (b - a) * torch.tensordot(weights, f(nodes), dims=1)


def odeint(
    f: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    t0: Union[float, Tensor],
    t1: Union[float, Tensor],
    phi: Iterable[Tensor] = (),
) -> Tensor:
    r"""Integrates a system of first-order ordinary differential equations (ODEs)

    .. math:: \frac{\mathrm{d} x}{\mathrm{d} t} = f_\phi(x, t) ,

    from :math:`t_0` to :math:`t_1` using the adaptive Dormand-Prince method. The
    output is the final state

    .. math:: x(t_1) = x_0 + \int_{t_0}^{t_1} f_\phi(x(t), t) ~ dt .

    Gradients are propagated through :math:`x_0`, :math:`t_0`, :math:`t_1` and
    :math:`\phi` via the adaptive checkpoint adjoint (ACA) method.

    References:
        | Neural Ordinary Differential Equations (Chen el al., 2018)
        | https://arxiv.org/abs/1806.07366

        | Adaptive Checkpoint Adjoint Method for Gradient Estimation in Neural ODE (Zhuang et al., 2020)
        | https://arxiv.org/abs/2006.02493

    Arguments:
        f: A system of first-order ODEs :math:`f_\phi`.
        x: The initial state :math:`x_0`.
        t0: The initial integration time :math:`t_0`.
        t1: The final integration time :math:`t_1`.
        phi: The parameters :math:`\phi` of :math:`f_\phi`.

    Example:
        >>> A = torch.randn(3, 3)
        >>> f = lambda x, t: x @ A
        >>> x0 = torch.randn(3)
        >>> x1 = odeint(f, x0, 0.0, 1.0)
        >>> x1
        tensor([-3.7454, -0.4140,  0.2677])
    """

    t0 = torch.as_tensor(t0).to(x)
    t1 = torch.as_tensor(t1).to(x)

    return AdaptiveCheckpointAdjoint.apply(f, x, t0, t1, *phi)


def dopri45(
    f: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    t: Tensor,
    dt: Tensor,
    error: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    r"""Applies one step of the Dormand-Prince method.

    Wikipedia:
        https://wikipedia.org/wiki/Dormand-Prince_method
    """

    k1 = dt * f(x, t)
    k2 = dt * f(x + 1 / 5 * k1, t + 1 / 5 * dt)
    k3 = dt * f(x + 3 / 40 * k1 + 9 / 40 * k2, t + 3 / 10 * dt)
    k4 = dt * f(x + 44 / 45 * k1 - 56 / 15 * k2 + 32 / 9 * k3, t + 4 / 5 * dt)
    k5 = dt * f(
        x + 19372 / 6561 * k1 - 25360 / 2187 * k2 + 64448 / 6561 * k3 - 212 / 729 * k4,
        t + 8 / 9 * dt,
    )
    k6 = dt * f(
        x
        + 9017 / 3168 * k1
        - 355 / 33 * k2
        + 46732 / 5247 * k3
        + 49 / 176 * k4
        - 5103 / 18656 * k5,
        t + dt,
    )
    x_next = (
        x
        + 35 / 384 * k1
        + 500 / 1113 * k3
        + 125 / 192 * k4
        - 2187 / 6784 * k5
        + 11 / 84 * k6
    )

    if not error:
        return x_next

    k7 = dt * f(x_next, t + dt)
    x_star = (
        x
        + 5179 / 57600 * k1
        + 7571 / 16695 * k3
        + 393 / 640 * k4
        - 92097 / 339200 * k5
        + 187 / 2100 * k6
        + 1 / 40 * k7
    )

    return x_next, abs(x_next - x_star)


class NestedTensor(tuple):
    r"""Creates an efficient data-structure to hold and perform basic operations on
    lists of tensors.
    """

    def __new__(cls, tensors: Iterable[Tensor] = ()) -> NestedTensor:
        return tuple.__new__(cls, tensors)

    def __add__(self, other: NestedTensor) -> NestedTensor:
        return NestedTensor(x + y for x, y in zip(self, other))

    def __sub__(self, other: NestedTensor) -> NestedTensor:
        return NestedTensor(x - y for x, y in zip(self, other))

    def __rmul__(self, factor: Tensor) -> NestedTensor:
        return NestedTensor(factor * x for x in self)

    def __abs__(self) -> NestedTensor:
        return NestedTensor(map(abs, self))


class AdaptiveCheckpointAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        f: Callable[[Tensor, Tensor], Tensor],
        x: Tensor,
        t0: Tensor,
        t1: Tensor,
        *phi: Tensor,
    ) -> Tensor:
        ctx.f = f
        ctx.save_for_backward(x, t0, t1, *phi)
        ctx.steps = []

        t, dt = t0, t1 - t0
        sign = torch.sign(dt)

        while sign * (t1 - t) > 0:
            dt = sign * torch.min(abs(dt), abs(t1 - t))

            while True:
                y, error = dopri45(f, x, t, dt, error=True)
                tolerance = 1e-6 + 1e-5 * torch.max(abs(x), abs(y))
                error = torch.max(error / tolerance).item() + 1e-6

                if error < 1.0:
                    x, t = y, t + dt
                    ctx.steps.append((x, t, dt))

                dt = dt * min(10.0, max(0.1, 0.9 / error ** (1 / 5)))

                if error < 1.0:
                    break

        return x

    @staticmethod
    def backward(ctx, grad_x: Tensor) -> Tuple[Tensor, ...]:
        f = ctx.f
        x0, t0, t1, *phi = ctx.saved_tensors
        x1, _, _ = ctx.steps[-1]

        # Final time
        if ctx.needs_input_grad[3]:
            grad_t1 = f(x1, t1) * grad_x
        else:
            grad_t1 = None

        # Adjoint
        grad_phi = tuple(map(torch.zeros_like, phi))

        def g(x: NestedTensor, t: Tensor) -> NestedTensor:
            x, grad_x, *_ = x

            with torch.enable_grad():
                x = x.detach().requires_grad_()
                dx = f(x, t)

            grad_x, *grad_phi = torch.autograd.grad(dx, (x, *phi), -grad_x, retain_graph=True)

            return NestedTensor((dx, grad_x, *grad_phi))

        for x, t, dt in reversed(ctx.steps):
            x = NestedTensor((x, grad_x, *grad_phi))
            x, grad_x, *grad_phi = dopri45(g, x, t, -dt)

        # Initial time
        if ctx.needs_input_grad[2]:
            grad_t0 = f(x0, t0) * grad_x
        else:
            grad_t0 = None

        return (None, grad_x, grad_t0, grad_t1, *grad_phi)
