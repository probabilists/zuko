r"""General purpose helpers."""

from __future__ import annotations

__all__ = [
    "Partial",
    "bisection",
    "broadcast",
    "gauss_legendre",
    "odeint",
    "unpack",
]

import math
import numpy as np
import torch
import torch.nn as nn

from functools import lru_cache
from torch import Size, Tensor
from torch.autograd.function import once_differentiable
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple, Union


class Partial(nn.Module):
    r"""A version of :class:`functools.partial` that is a :class:`torch.nn.Module`.

    Arguments:
        f: An arbitrary callable. If `f` is a module, it is registered as a submodule.
        args: The positional arguments passed to `f`.
        buffer: Whether tensor arguments are registered as buffers or parameters.
        kwargs: The keyword arguments passed to `f`.

    Examples:
        >>> increment = Partial(torch.add, torch.tensor(1.0), buffer=True)
        >>> increment(torch.arange(3))
        tensor([1., 2., 3.])

        >>> weight = torch.randn((5, 3))
        >>> linear = Partial(torch.nn.functional.linear, weight=weight)
        >>> x = torch.rand(2, 3)
        >>> linear(x)
        tensor([[-0.1364, -0.4034,  0.1887, -0.2045, -0.0151],
                [-2.0380, -1.5081, -0.4816,  0.0323, -0.7941]], grad_fn=<MmBackward0>)

        >>> f = torch.distributions.Normal
        >>> loc, scale = torch.zeros(3), torch.ones(3)
        >>> dist = Partial(f, loc, scale, buffer=True)
        >>> dist()
        Normal(loc: torch.Size([3]), scale: torch.Size([3]))
        >>> dist().sample()
        tensor([ 0.1227,  0.1494, -0.6709])
    """

    def __init__(
        self,
        f: Callable,
        /,
        *args: Any,
        buffer: bool = False,
        **kwargs: Any,
    ):
        super().__init__()

        self.f = f

        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                if buffer:
                    self.register_buffer(f"_{i}", arg)
                else:
                    self.register_parameter(f"_{i}", nn.Parameter(arg))
            else:
                setattr(self, f"_{i}", arg)

        self._nargs = len(args)

        for key, arg in kwargs.items():
            if torch.is_tensor(arg):
                if buffer:
                    self.register_buffer(key, arg)
                else:
                    self.register_parameter(key, nn.Parameter(arg))
            else:
                setattr(self, key, arg)

        self._keys = list(kwargs.keys())

    @property
    def args(self) -> Sequence[Any]:
        return [getattr(self, f"_{i}") for i in range(self._nargs)]

    @property
    def kwargs(self) -> Dict[str, Any]:
        return {key: getattr(self, key) for key in self._keys}

    def extra_repr(self) -> str:
        if isinstance(self.f, nn.Module):
            return ""
        else:
            return f"(f): {self.f}"

    def forward(self, *args, **kwargs) -> Any:
        r"""
        Returns:
            :py:`self.f(*self.args, *args, **self.kwargs, **kwargs)`
        """

        return self.f(
            *self.args,
            *args,
            **self.kwargs,
            **kwargs,
        )


def bisection(
    f: Callable[[Tensor], Tensor],
    y: Tensor,
    a: Union[float, Tensor],
    b: Union[float, Tensor],
    n: int = 16,
    phi: Iterable[Tensor] = (),
) -> Tensor:
    r"""Applies the bisection method to find :math:`x` between the bounds :math:`a`
    and :math:`b` such that :math:`f_\phi(x)` is close to :math:`y`.

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

    Returns:
        The solution :math:`x`.

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
        for _ in range(n):
            c = (a + b) / 2

            mask = f(c) < y

            a = torch.where(mask, c, a)
            b = torch.where(mask, b, c)

        x = (a + b) / 2

        ctx.f = f
        ctx.save_for_backward(x, *phi)

        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_x: Tensor) -> Tuple[Tensor, ...]:
        f = ctx.f
        x, *phi = ctx.saved_tensors

        with torch.enable_grad():
            x = x.detach().requires_grad_()
            y = f(x)

        (jacobian,) = torch.autograd.grad(y, x, torch.ones_like(y), retain_graph=True)
        grad_y = grad_x / jacobian

        if phi:
            grad_phi = torch.autograd.grad(y, phi, -grad_y, retain_graph=True, allow_unused=True)
        else:
            grad_phi = ()

        return (None, grad_y, None, None, None, *grad_phi)


def broadcast(*tensors: Tensor, ignore: Union[int, Sequence[int]] = 0) -> List[Tensor]:
    r"""Broadcasts tensors together.

    The term broadcasting describes how PyTorch treats tensors with different shapes
    during arithmetic operations. In short, if possible, dimensions that have
    different sizes are expanded (without making copies) to be compatible.

    Arguments:
        tensors: The tensors to broadcast.
        ignore: The number(s) of dimensions not to broadcast.

    Returns:
        The broadcasted tensors.

    Example:
        >>> x = torch.rand(3, 1, 2)
        >>> y = torch.rand(4, 5)
        >>> x, y = broadcast(x, y, ignore=1)
        >>> x.shape
        torch.Size([3, 4, 2])
        >>> y.shape
        torch.Size([3, 4, 5])
    """

    if isinstance(ignore, int):
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

    Returns:
        The definite integral estimation.

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
                area = GaussLegendre.quadrature(f, a, b, n)

            grad_phi = torch.autograd.grad(
                area, phi, grad_area, create_graph=True, allow_unused=True
            )
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

        kwargs.setdefault("dtype", torch.get_default_dtype())

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
    x: Union[Tensor, Sequence[Tensor]],
    t0: Union[float, Tensor],
    t1: Union[float, Tensor],
    phi: Iterable[Tensor] = (),
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> Union[Tensor, Sequence[Tensor]]:
    r"""Integrates a system of first-order ordinary differential equations (ODEs)

    .. math:: \frac{dx}{dt} = f_\phi(t, x) ,

    from :math:`t_0` to :math:`t_1` using the adaptive Dormand-Prince method. The
    output is the final state

    .. math:: x(t_1) = x_0 + \int_{t_0}^{t_1} f_\phi(t, x(t)) ~ dt .

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
        atol: The absolute tolerance.
        rtol: The relative tolerance.

    Returns:
        The final state :math:`x(t_1)`.

    Example:
        >>> A = torch.randn(3, 3)
        >>> f = lambda t, x: x @ A
        >>> x0 = torch.randn(3)
        >>> x1 = odeint(f, x0, 0.0, 1.0)
        >>> x1
        tensor([-1.4596,  0.5008,  1.5828])
    """

    settings = (atol, rtol, torch.is_grad_enabled())

    if torch.is_tensor(x):
        x0 = x
        g = f
    else:
        shapes = [y.shape for y in x]

        def pack(x: Iterable[Tensor]) -> Tensor:
            return torch.cat([y.flatten() for y in x])

        x0 = pack(x)
        g = lambda t, x: pack(f(t, *unpack(x, shapes)))

    t0 = torch.as_tensor(t0, dtype=x0.dtype, device=x0.device)
    t1 = torch.as_tensor(t1, dtype=x0.dtype, device=x0.device)

    assert not t0.shape and not t1.shape, "'t0' and 't1' must be scalars"

    x1 = AdaptiveCheckpointAdjoint.apply(settings, g, x0, t0, t1, *phi)

    if torch.is_tensor(x):
        return x1
    else:
        return unpack(x1, shapes)


# fmt: off
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

    k1 = dt * f(t, x)
    k2 = dt * f(t + 1 / 5 * dt, x + 1 / 5 * k1)
    k3 = dt * f(t + 3 / 10 * dt, x + 3 / 40 * k1 + 9 / 40 * k2)
    k4 = dt * f(t + 4 / 5 * dt, x + 44 / 45 * k1 - 56 / 15 * k2 + 32 / 9 * k3)
    k5 = dt * f(
        t + 8 / 9 * dt,
        x + 19372 / 6561 * k1 - 25360 / 2187 * k2 + 64448 / 6561 * k3 - 212 / 729 * k4,
    )
    k6 = dt * f(
        t + dt,
        x
        + 9017 / 3168 * k1
        - 355 / 33 * k2
        + 46732 / 5247 * k3
        + 49 / 176 * k4
        - 5103 / 18656 * k5,
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

    k7 = dt * f(t + dt, x_next)
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
# fmt: on


class NestedTensor(tuple):
    r"""Creates an efficient data structure to hold and perform basic operations
    on sequences of tensors.
    """

    def __add__(self, other: NestedTensor) -> NestedTensor:
        return NestedTensor(x + y for x, y in zip(self, other))

    def __sub__(self, other: NestedTensor) -> NestedTensor:
        return NestedTensor(x - y for x, y in zip(self, other))

    def __rmul__(self, other: Tensor) -> NestedTensor:
        return NestedTensor(other * x for x in self)


class AdaptiveCheckpointAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        settings: Tuple[float, float, bool],
        f: Callable[[Tensor, Tensor], Tensor],
        x: Tensor,
        t0: Tensor,
        t1: Tensor,
        *phi: Tensor,
    ) -> Tensor:
        atol, rtol, grad_enabled = settings

        ctx.f = f
        ctx.save_for_backward(x, t0, t1, *phi)
        ctx.steps = []

        t, dt = t0, t1 - t0
        sign = torch.sign(dt)

        while sign * (t1 - t) > 0:
            dt = sign * torch.min(abs(dt), abs(t1 - t))

            while True:
                y, error = dopri45(f, x, t, dt, error=True)
                tolerance = atol + rtol * torch.max(abs(x), abs(y))
                error = torch.max(error / tolerance).clip(min=1e-9).item()

                if error < 1.0:
                    x, t = y, t + dt

                    if grad_enabled:
                        ctx.steps.append((x, t, dt))

                dt = dt * min(10.0, max(0.1, 0.9 / error ** (1 / 5)))

                if error < 1.0:
                    break

        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_x: Tensor) -> Tuple[Tensor, ...]:
        f = ctx.f
        x0, t0, t1, *phi = ctx.saved_tensors
        x1, _, _ = ctx.steps[-1]

        # Final time
        if ctx.needs_input_grad[4]:
            grad_t1 = torch.sum(f(t1, x1) * grad_x)
        else:
            grad_t1 = None

        # Adjoint
        grad_phi = map(torch.zeros_like, phi)

        def g(t: Tensor, x_aug: NestedTensor) -> NestedTensor:
            x, grad_x, *_ = x_aug

            with torch.enable_grad():
                x = x.detach().requires_grad_()
                dx = f(t, x)

            grad_x, *grad_phi = torch.autograd.grad(dx, (x, *phi), -grad_x)

            return NestedTensor((dx, grad_x, *grad_phi))

        for x, t, dt in reversed(ctx.steps):
            x_aug = NestedTensor((x, grad_x, *grad_phi))
            _, grad_x, *grad_phi = dopri45(g, x_aug, t, -dt)

        # Initial time
        if ctx.needs_input_grad[3]:
            grad_t0 = torch.sum(f(t0, x0) * grad_x)
        else:
            grad_t0 = None

        return (None, None, grad_x, grad_t0, grad_t1, *grad_phi)


def unpack(x: Tensor, shapes: Sequence[Size]) -> Sequence[Tensor]:
    r"""Unpacks a packed tensor.

    Arguments:
        x: A packed tensor, with shape :math:`(*, D)`.
        shapes: A sequence of shapes :math:`S_i`, corresponding to the total number of
            elements :math:`D`.

    Returns:
        The unpacked tensors, with shapes :math:`(*, S_i)`.

    Example:
        >>> x = torch.randn(26)
        >>> y, z = unpack(x, ((1, 2, 3), (4, 5)))
        >>> y.shape
        torch.Size([1, 2, 3])
        >>> z.shape
        torch.Size([4, 5])
    """

    sizes = [math.prod(s) for s in shapes]

    x = x.split(sizes, -1)
    x = (y.unflatten(-1, (*s, 1)) for y, s in zip(x, shapes))
    x = (y.squeeze(-1) for y in x)

    return tuple(x)
