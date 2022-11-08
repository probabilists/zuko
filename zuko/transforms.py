r"""Parameterizable transformations."""

import math
import torch
import torch.nn.functional as F

from torch import Tensor, LongTensor
from torch.distributions import *
from torch.distributions import constraints
from typing import *

from .utils import bisection, broadcast, gauss_legendre


class IdentityTransform(Transform):
    r"""Creates a transformation :math:`f(x) = x`."""

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, IdentityTransform)

    def _call(self, x: Tensor) -> Tensor:
        return x

    def _inverse(self, y: Tensor) -> Tensor:
        return y

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.zeros_like(x)


class CosTransform(Transform):
    r"""Creates a transformation :math:`f(x) = -\cos(x)`."""

    domain = constraints.interval(0, math.pi)
    codomain = constraints.interval(-1, 1)
    bijective = True
    sign = +1

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, CosTransform)

    def _call(self, x: Tensor) -> Tensor:
        return -x.cos()

    def _inverse(self, y: Tensor) -> Tensor:
        return (-y).acos()

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return x.sin().abs().log()


class SinTransform(Transform):
    r"""Creates a transformation :math:`f(x) = \sin(x)`."""

    domain = constraints.interval(-math.pi / 2, math.pi / 2)
    codomain = constraints.interval(-1, 1)
    bijective = True
    sign = +1

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, SinTransform)

    def _call(self, x: Tensor) -> Tensor:
        return x.sin()

    def _inverse(self, y: Tensor) -> Tensor:
        return y.asin()

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return x.cos().abs().log()


class SoftclipTransform(Transform):
    r"""Creates a transform that maps :math:`\mathbb{R}` to the inverval :math:`[-B, B]`.

    .. math:: f(x) = \frac{x}{1 + \left| \frac{x}{B} \right|}

    Arguments:
        bound: The codomain bound :math:`B`.
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(self, bound: float = 5.0, **kwargs):
        super().__init__(**kwargs)

        self.bound = bound

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(bound={self.bound})'

    def _call(self, x: Tensor) -> Tensor:
        return x / (1 + abs(x / self.bound))

    def _inverse(self, y: Tensor) -> Tensor:
        return y / (1 - abs(y / self.bound))

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return -2 * torch.log1p(abs(x / self.bound))


class MonotonicAffineTransform(Transform):
    r"""Creates a transformation :math:`f(x) = \alpha x + \beta`.

    Arguments:
        shift: The shift term :math:`\beta`, with shape :math:`(*,)`.
        scale: The unconstrained scale factor :math:`\alpha`, with shape :math:`(*,)`.
        slope: The minimum slope of the transformation.
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(
        self,
        shift: Tensor,
        scale: Tensor,
        slope: float = 1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.shift = shift
        self.log_scale = scale / (1 + abs(scale / math.log(slope)))
        self.scale = self.log_scale.exp()

    def _call(self, x: Tensor) -> Tensor:
        return x * self.scale + self.shift

    def _inverse(self, y: Tensor) -> Tensor:
        return (y - self.shift) / self.scale

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return self.log_scale.expand(x.shape)


class MonotonicRQSTransform(Transform):
    r"""Creates a monotonic rational-quadratic spline (RQS) transformation.

    References:
        | Neural Spline Flows (Durkan et al., 2019)
        | https://arxiv.org/abs/1906.04032

    Arguments:
        widths: The unconstrained bin widths, with shape :math:`(*, K)`.
        heights: The unconstrained bin heights, with shape :math:`(*, K)`.
        derivatives: The unconstrained knot derivatives, with shape :math:`(*, K - 1)`.
        bound: The spline's (co)domain bound :math:`B`.
        slope: The minimum slope of the transformation.
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(
        self,
        widths: Tensor,
        heights: Tensor,
        derivatives: Tensor,
        bound: float = 5.0,
        slope: float = 1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        widths = widths / (1 + abs(2 * widths / math.log(slope)))
        heights = heights / (1 + abs(2 * heights / math.log(slope)))
        derivatives = derivatives / (1 + abs(derivatives / math.log(slope)))

        widths = 2 * F.softmax(widths, dim=-1)
        heights = 2 * F.softmax(heights, dim=-1)
        derivatives = derivatives.exp()

        self.horizontal = bound * torch.cumsum(F.pad(widths, (1, 0), value=-1), dim=-1)
        self.vertical = bound * torch.cumsum(F.pad(heights, (1, 0), value=-1), dim=-1)
        self.derivatives = F.pad(derivatives, (1, 1), value=1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(bins={self.bins})'

    @property
    def bins(self) -> int:
        return self.horizontal.shape[-1] - 1

    def bin(self, k: LongTensor) -> Tuple[Tensor, ...]:
        mask = torch.logical_and(0 <= k, k < self.bins)

        k = k % self.bins
        k0_k1 = torch.stack((k, k + 1))

        k0_k1, hs, vs, ds = broadcast(
            k0_k1[..., None],
            self.horizontal,
            self.vertical,
            self.derivatives,
            ignore=1,
        )

        x0, x1 = hs.gather(-1, k0_k1).squeeze(dim=-1)
        y0, y1 = vs.gather(-1, k0_k1).squeeze(dim=-1)
        d0, d1 = ds.gather(-1, k0_k1).squeeze(dim=-1)

        s = (y1 - y0) / (x1 - x0)

        return mask, x0, x1, y0, y1, d0, d1, s

    @staticmethod
    def searchsorted(seq: Tensor, value: Tensor) -> LongTensor:
        return torch.searchsorted(seq, value[..., None]).squeeze(dim=-1)

    def _call(self, x: Tensor) -> Tensor:
        k = self.searchsorted(self.horizontal, x) - 1
        mask, x0, x1, y0, y1, d0, d1, s = self.bin(k)

        z = mask * (x - x0) / (x1 - x0)

        y = y0 + (y1 - y0) * (s * z**2 + d0 * z * (1 - z)) / (
            s + (d0 + d1 - 2 * s) * z * (1 - z)
        )

        return torch.where(mask, y, x)

    def _inverse(self, y: Tensor) -> Tensor:
        k = self.searchsorted(self.vertical, y) - 1
        mask, x0, x1, y0, y1, d0, d1, s = self.bin(k)

        y_ = mask * (y - y0)

        a = (y1 - y0) * (s - d0) + y_ * (d0 + d1 - 2 * s)
        b = (y1 - y0) * d0 - y_ * (d0 + d1 - 2 * s)
        c = -s * y_

        z = 2 * c / (-b - (b**2 - 4 * a * c).sqrt())

        x = x0 + z * (x1 - x0)

        return torch.where(mask, x, y)

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        k = self.searchsorted(self.horizontal, x) - 1
        mask, x0, x1, y0, y1, d0, d1, s = self.bin(k)

        z = mask * (x - x0) / (x1 - x0)

        jacobian = (
            s**2
            * (2 * s * z * (1 - z) + d0 * (1 - z) ** 2 + d1 * z**2)
            / (s + (d0 + d1 - 2 * s) * z * (1 - z)) ** 2
        )

        return mask * jacobian.log()


class MonotonicTransform(Transform):
    r"""Creates a transformation from a monotonic univariate function :math:`f(x)`.

    The inverse function :math:`f^{-1}` is approximated using the bisection method.

    Arguments:
        f: A monotonic univariate function :math:`f(x)`.
        bound: The domain bound :math:`B`.
        eps: The absolute tolerance for the inverse transformation.
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(
        self,
        f: Callable[[Tensor], Tensor],
        bound: float = 5.0,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.f = f
        self.bound = bound
        self.eps = eps

    def _call(self, x: Tensor) -> Tensor:
        return self.f(x)

    def _inverse(self, y: Tensor) -> Tensor:
        return bisection(
            f=lambda x: self.f(x) - y,
            a=torch.full_like(y, -self.bound),
            b=torch.full_like(y, self.bound),
            n=math.ceil(math.log2(2 * self.bound / self.eps)),
        )

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.log(
            torch.autograd.functional.jacobian(
                func=lambda x: self.f(x).sum(),
                inputs=x,
                create_graph=True,
            )
        )


class UnconstrainedMonotonicTransform(MonotonicTransform):
    r"""Creates a monotonic transformation :math:`f(x)` by integrating a positive
    univariate function :math:`g(x)`.

    .. math:: f(x) = \int_0^x g(u) ~ du + C

    The definite integral is estimated by a :math:`n`-point Gauss-Legendre quadrature.

    Arguments:
        g: A positive univariate function :math:`g(x)`.
        C: The integration constant :math:`C`.
        n: The number of points :math:`n` for the quadrature.
        kwargs: Keyword arguments passed to :class:`MonotonicTransform`.
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(
        self,
        g: Callable[[Tensor], Tensor],
        C: Tensor,
        n: int = 16,
        **kwargs,
    ):
        super().__init__(self.f, **kwargs)

        self.g = g
        self.C = C
        self.n = n

    def f(self, x: Tensor) -> Tensor:
        return gauss_legendre(
            f=self.g,
            a=torch.zeros_like(x),
            b=x,
            n=self.n,
        ) + self.C

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return self.g(x).log()


class SOSPolynomialTransform(UnconstrainedMonotonicTransform):
    r"""Creates a sum-of-squares (SOS) polynomial transformation.

    The transformation :math:`f(x)` is expressed as the primitive integral of the
    sum of :math:`K` squared polynomials of degree :math:`L`.

    .. math:: f(x) = \int_0^x \sum_{i = 1}^K
        \left( 1 + \sum_{j = 0}^L a_{i,j} ~ u^j \right)^2 ~ du + C

    References:
        | Sum-of-Squares Polynomial Flow (Jaini et al., 2019)
        | https://arxiv.org/abs/1905.02325

    Arguments:
        a: The polynomial coefficients :math:`a`, with shape :math:`(*, K, L + 1)`.
        C: The integration constant :math:`C`.
        kwargs: Keyword arguments passed to :class:`UnconstrainedMonotonicTransform`.
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(self, a: Tensor, C: Tensor, **kwargs):
        super().__init__(self.g, C, a.shape[-1], **kwargs)

        self.a = a
        self.i = torch.arange(a.shape[-1]).to(a.device)

    def g(self, x: Tensor) -> Tensor:
        x = x / self.bound
        x = x[..., None] ** self.i
        p = 1 + self.a @ x[..., None]

        return p.squeeze(dim=-1).square().sum(dim=-1)


class AutoregressiveTransform(Transform):
    r"""Transform via an autoregressive mapping.

    Arguments:
        meta: A meta function which returns a transformation :math:`f`.
        passes: The number of passes for the inverse transformation.
    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(
        self,
        meta: Callable[[Tensor], Transform],
        passes: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.meta = meta
        self.passes = passes

        self._cache = None, None

    def _call(self, x: Tensor) -> Tensor:
        _x, _f = self._cache

        if x is _x:
            f = _f
        else:
            f = self.meta(x)

        self._cache = x, f

        return f(x)

    def _inverse(self, y: Tensor) -> Tensor:
        x = torch.zeros_like(y)
        for _ in range(self.passes):
            x = self.meta(x).inv(y)

        return x

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        _x, _f = self._cache

        if x is _x:
            f = _f
        else:
            f = self.meta(x)

        self._cache = x, f

        return f.log_abs_det_jacobian(x, y).sum(dim=-1)


class PermutationTransform(Transform):
    r"""Creates a transformation that permutes the elements.

    Arguments:
        order: The permutation order, with shape :math:`(*, D)`.
    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, order: LongTensor, **kwargs):
        super().__init__(**kwargs)

        self.order = order
        self.inverse = torch.argsort(order)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.order.tolist()})'

    def _call(self, x: Tensor) -> Tensor:
        return x[..., self.order]

    def _inverse(self, y: Tensor) -> Tensor:
        return y[..., self.inverse]

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return x.new_zeros(x.shape[:-1])
