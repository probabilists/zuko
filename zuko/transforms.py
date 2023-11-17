r"""Parameterizable transformations."""

__all__ = [
    'ComposedTransform',
    'DependentTransform',
    'IdentityTransform',
    'CosTransform',
    'SinTransform',
    'SoftclipTransform',
    'CircularShiftTransform',
    'SignedPowerTransform',
    'MonotonicAffineTransform',
    'MonotonicRQSTransform',
    'MonotonicTransform',
    'GaussianizationTransform',
    'UnconstrainedMonotonicTransform',
    'SOSPolynomialTransform',
    'AutoregressiveTransform',
    'CouplingTransform',
    'FreeFormJacobianTransform',
    'PermutationTransform',
    'RotationTransform',
    'LULinearTransform',
]

import math
import torch
import torch.nn.functional as F

from textwrap import indent
from torch import Tensor, BoolTensor, LongTensor, Size
from torch.distributions import constraints
from torch.distributions.transforms import *
from torch.distributions.utils import _sum_rightmost
from typing import *

from .utils import bisection, broadcast, gauss_legendre, odeint


torch.distributions.transforms._InverseTransform.__name__ = 'Inverse'


def _call_and_ladj(self, x: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Returns both the transformed value and the log absolute determinant of the
    transformation's Jacobian."""

    y = self.__call__(x)
    ladj = self.log_abs_det_jacobian(x, y)

    return y, ladj


Transform.call_and_ladj = _call_and_ladj


class ComposedTransform(Transform):
    r"""Creates a transformation :math:`f(x) = f_n \circ \dots \circ f_0(x)`.

    Optimized version of :class:`torch.distributions.transforms.ComposeTransform`.

    Arguments:
        transforms: A sequence of transformations :math:`f_i`.
    """

    def __init__(self, *transforms: Transform, **kwargs):
        super().__init__(**kwargs)

        assert transforms, "'transforms' cannot be empty"

        event_dim = 0

        for t in reversed(transforms):
            event_dim = t.domain.event_dim + max(event_dim - t.codomain.event_dim, 0)

        self.domain_dim = event_dim

        for t in transforms:
            event_dim += t.codomain.event_dim - t.domain.event_dim

        self.codomain_dim = event_dim
        self.transforms = transforms

    def __repr__(self) -> str:
        lines = [f'({i}): {t}' for i, t in enumerate(self.transforms)]
        lines = indent('\n'.join(lines), '  ')

        return f'{self.__class__.__name__}(\n' + lines + '\n)'

    @property
    def domain(self) -> constraints.Constraint:
        domain = self.transforms[0].domain
        reinterpreted = self.domain_dim - domain.event_dim

        if reinterpreted > 0:
            return constraints.independent(domain, reinterpreted)
        else:
            return domain

    @property
    def codomain(self) -> constraints.Constraint:
        codomain = self.transforms[-1].codomain
        reinterpreted = self.codomain_dim - codomain.event_dim

        if reinterpreted > 0:
            return constraints.independent(codomain, reinterpreted)
        else:
            return codomain

    @property
    def bijective(self) -> bool:
        return all(t.bijective for t in self.transforms)

    def _call(self, x: Tensor) -> Tensor:
        for t in self.transforms:
            x = t(x)
        return x

    @property
    def inv(self) -> Transform:
        new = self.__new__(ComposedTransform)
        new.transforms = [t.inv for t in reversed(self.transforms)]
        new.domain_dim = self.codomain_dim
        new.codomain_dim = self.domain_dim

        Transform.__init__(new)

        return new

    def _inverse(self, y: Tensor) -> Tensor:
        for t in reversed(self.transforms):
            y = t.inv(y)
        return y

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        _, ladj = self.call_and_ladj(x)
        return ladj

    def call_and_ladj(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        event_dim = self.domain_dim
        acc = 0

        for t in self.transforms:
            x, ladj = t.call_and_ladj(x)
            acc = acc + _sum_rightmost(ladj, event_dim - t.domain.event_dim)
            event_dim += t.codomain.event_dim - t.domain.event_dim

        return x, acc

    def forward_shape(self, shape: Size) -> Size:
        for t in self.transforms:
            shape = t.forward_shape(shape)
        return shape

    def inverse_shape(self, shape: Size) -> Size:
        for t in reversed(self.transforms):
            shape = t.inverse_shape(shape)
        return shape


class DependentTransform(Transform):
    r"""Wraps a base transformation to treat right-most dimensions as dependent.

    Optimized version of :class:`torch.distributions.transforms.IndependentTransform`.

    Arguments:
        base: The base transformation.
        reinterpreted: The number of dimensions to treat as dependent.
    """

    def __init__(self, base: Transform, reinterpreted: int, **kwargs):
        super().__init__(**kwargs)

        self.base = base
        self.reinterpreted = reinterpreted

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.base}, {self.reinterpreted})'

    @property
    def domain(self) -> constraints.Constraint:
        return constraints.independent(self.base.domain, self.reinterpreted)

    @property
    def codomain(self) -> constraints.Constraint:
        return constraints.independent(self.base.codomain, self.reinterpreted)

    @property
    def bijective(self) -> bool:
        return self.base.bijective

    def _call(self, x: Tensor) -> Tensor:
        return self.base(x)

    @property
    def inv(self) -> Transform:
        return DependentTransform(self.base.inv, self.reinterpreted)

    def _inverse(self, y: Tensor) -> Tensor:
        return self.base.inv(y)

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        ladj = self.base.log_abs_det_jacobian(x, y)
        ladj = _sum_rightmost(ladj, self.reinterpreted)

        return ladj

    def call_and_ladj(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        y, ladj = self.base.call_and_ladj(x)
        ladj = ladj = _sum_rightmost(ladj, self.reinterpreted)

        return y, ladj

    def forward_shape(self, shape: Size) -> Size:
        return self.base.forward_shape(shape)

    def inverse_shape(self, shape: Size) -> Size:
        return self.base.inverse_shape(shape)


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
    r"""Creates a transformation that maps :math:`\mathbb{R}` to the interval
    :math:`[-B, B]`.

    .. math:: f(x) = \frac{x}{1 + \left| \frac{x}{B} \right|}

    Arguments:
        bound: The codomain bound :math:`B`.
    """

    bijective = True
    sign = +1

    def __init__(self, bound: float = 1.0, **kwargs):
        super().__init__(**kwargs)

        self.bound = bound
        self.domain = constraints.real
        self.codomain = constraints.interval(-bound, bound)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(bound={self.bound})'

    def _call(self, x: Tensor) -> Tensor:
        return x / (1 + abs(x / self.bound))

    def _inverse(self, y: Tensor) -> Tensor:
        return y / (1 - abs(y / self.bound))

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return -2 * torch.log1p(abs(x / self.bound))


class CircularShiftTransform(Transform):
    r"""Creates a transformation that circularly shifts the interval :math:`[-B, B]`.

    .. math:: f(x) = (x \bmod 2B) - B

    Note:
        This transformation is only bijective over its domain :math:`[-B, B]` as
        :math:`f(x) = f(x + 2kB)` for all :math:`k \in \mathbb{Z}`.

    Arguments:
        bound: The domain bound :math:`B`.
    """

    bijective = True

    def __init__(self, bound: float = 1.0, **kwargs):
        super().__init__(**kwargs)

        self.bound = bound
        self.domain = constraints.interval(-bound, bound)
        self.codomain = constraints.interval(-bound, bound)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(bound={self.bound})'

    def _call(self, x: Tensor) -> Tensor:
        return torch.remainder(x, 2 * self.bound) - self.bound

    def _inverse(self, y: Tensor) -> Tensor:
        return torch.remainder(y, 2 * self.bound) - self.bound

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.zeros_like(x)


class SignedPowerTransform(Transform):
    r"""Creates a transformation :math:`f(x) = \mathrm{sign}(x) |x|^{\exp(\alpha)}`.

    Arguments:
        alpha: The unconstrained exponent :math:`\alpha`, with shape :math:`(*,)`.
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(self, alpha: Tensor, **kwargs):
        super().__init__(**kwargs)

        self.alpha = alpha

    def _call(self, x: Tensor) -> Tensor:
        return x * abs(x) ** torch.expm1(self.alpha)

    def _inverse(self, y: Tensor) -> Tensor:
        return y * abs(y) ** torch.expm1(-self.alpha)

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return self.alpha + torch.expm1(self.alpha) * torch.log(abs(x))


class MonotonicAffineTransform(Transform):
    r"""Creates a transformation :math:`f(x) = \exp(a) x + b`.

    Arguments:
        shift: The shift term :math:`b`, with shape :math:`(*,)`.
        scale: The unconstrained scale factor :math:`a`, with shape :math:`(*,)`.
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
        slope: float = 1e-4,
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
        slope: float = 1e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)

        widths = widths / (1 + abs(2 * widths / math.log(slope)))
        heights = heights / (1 + abs(2 * heights / math.log(slope)))
        derivatives = derivatives / (1 + abs(derivatives / math.log(slope)))

        widths = F.pad(F.softmax(widths, dim=-1), (1, 0), value=0)
        heights = F.pad(F.softmax(heights, dim=-1), (1, 0), value=0)
        derivatives = F.pad(derivatives, (1, 1), value=0)

        self.horizontal = bound * (2 * torch.cumsum(widths, dim=-1) - 1)
        self.vertical = bound * (2 * torch.cumsum(heights, dim=-1) - 1)
        self.derivatives = torch.exp(derivatives)

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
        seq, value = broadcast(seq, value.unsqueeze(dim=-1), ignore=1)
        seq = seq.contiguous()

        return torch.searchsorted(seq, value).squeeze(dim=-1)

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
        _, ladj = self.call_and_ladj(x)
        return ladj

    def call_and_ladj(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        k = self.searchsorted(self.horizontal, x) - 1
        mask, x0, x1, y0, y1, d0, d1, s = self.bin(k)

        z = mask * (x - x0) / (x1 - x0)

        y = y0 + (y1 - y0) * (s * z**2 + d0 * z * (1 - z)) / (
            s + (d0 + d1 - 2 * s) * z * (1 - z)
        )

        jacobian = (
            s**2
            * (2 * s * z * (1 - z) + d0 * (1 - z) ** 2 + d1 * z**2)
            / (s + (d0 + d1 - 2 * s) * z * (1 - z)) ** 2
        )

        return torch.where(mask, y, x), mask * jacobian.log()


class MonotonicTransform(Transform):
    r"""Creates a transformation from a monotonic univariate function :math:`f_\phi(x)`.

    The inverse function :math:`f_\phi^{-1}` is approximated using the bisection method.

    Arguments:
        f: A monotonic univariate function :math:`f_\phi`.
        phi: The parameters :math:`\phi` of :math:`f_\phi`.
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
        phi: Iterable[Tensor] = (),
        bound: float = 10.0,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.f = f
        self.phi = tuple(filter(lambda p: p.requires_grad, phi))
        self.bound = bound
        self.eps = eps

    def _call(self, x: Tensor) -> Tensor:
        return self.f(x)

    def _inverse(self, y: Tensor) -> Tensor:
        return bisection(
            f=self.f,
            y=y,
            a=torch.full_like(y, -self.bound),
            b=torch.full_like(y, self.bound),
            n=math.ceil(math.log2(2 * self.bound / self.eps)),
            phi=self.phi,
        )

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        _, ladj = self.call_and_ladj(x)
        return ladj

    def call_and_ladj(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        with torch.enable_grad():
            x = x.clone().requires_grad_()
            y = self.f(x)

        jacobian = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]

        return y, jacobian.log()


class GaussianizationTransform(MonotonicTransform):
    r"""Creates a gaussianization transformation.

    .. math:: f(x) = \Phi^{-1}
        \left( \frac{1}{K} \sum_{i=1}^K \Phi(\exp(a_i) x + b_i) \right)

    where :math:`\Phi` is the cumulative distribution function (CDF) of the standard
    normal :math:`\mathcal{N}(0, 1)`.

    References:
        | Gaussianization (Chen et al., 2000)
        | https://papers.nips.cc/paper/1856-gaussianization

    Arguments:
        shift: The shift terms :math:`b`, with shape :math:`(*, K)`.
        scale: The unconstrained scale factors :math:`a`, with shape :math:`(*, K)`.
        kwargs: Keyword arguments passed to :class:`MonotonicTransform`.
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(
        self,
        shift: Tensor,
        scale: Tensor,
        **kwargs,
    ):
        super().__init__(self.f, phi=(shift, scale), **kwargs)

        self.shift = shift
        self.scale = torch.exp(scale)

    def f(self, x: Tensor) -> Tensor:
        x = x[..., None] * self.scale + self.shift
        x = torch.erf(x / math.sqrt(2))
        x = torch.mean(x, dim=-1) * (1 - 1e-6)
        x = torch.erfinv(x) * math.sqrt(2)

        return x


class UnconstrainedMonotonicTransform(MonotonicTransform):
    r"""Creates a monotonic transformation :math:`f(x)` by integrating a positive
    univariate function :math:`g(x)`.

    .. math:: f(x) = \int_0^x g(u) ~ du + C

    The definite integral is estimated by a :math:`n`-point Gauss-Legendre quadrature.

    Arguments:
        g: A positive univariate function :math:`g`.
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
        n: int = 32,
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
            phi=self.phi,
        ) + self.C

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return self.g(x).log()

    def call_and_ladj(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.f(x), self.g(x).log()


class SOSPolynomialTransform(UnconstrainedMonotonicTransform):
    r"""Creates a sum-of-squares (SOS) polynomial transformation.

    The transformation :math:`f(x)` is expressed as the primitive integral of the
    sum of :math:`K` squared polynomials of degree :math:`L`.

    .. math:: f(x) = \int_0^x \frac{1}{K} \sum_{i = 1}^K
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
        super().__init__(self.g, C, phi=(a,), n=a.shape[-1], **kwargs)

        self.a = a
        self.i = torch.arange(a.shape[-1], device=a.device)

    def g(self, x: Tensor) -> Tensor:
        x = x / self.bound
        x = x[..., None] ** self.i
        p = 1 + self.a @ x[..., None]

        return p.squeeze(dim=-1).square().mean(dim=-1)


class AutoregressiveTransform(Transform):
    r"""Transform via an autoregressive scheme.

    .. math:: y_i = f(x_i | x_{<i})

    Arguments:
        meta: A function which returns a transformation :math:`f` given :math:`x`.
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

    def _call(self, x: Tensor) -> Tensor:
        return self.meta(x)(x)

    def _inverse(self, y: Tensor) -> Tensor:
        x = torch.zeros_like(y)
        for _ in range(self.passes):
            x = self.meta(x).inv(y)

        return x

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return self.meta(x).log_abs_det_jacobian(x, y)

    def call_and_ladj(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        y, ladj = self.meta(x).call_and_ladj(x)
        return y, ladj


class CouplingTransform(Transform):
    r"""Transform via a coupling scheme.

    .. math::
        y_a & = x_a \\
        y_b & = f(x_b | x_a)

    Arguments:
        meta: A function which returns a transformation :math:`f` given :math:`x_a`.
        mask: A coupling mask defining the split :math:`x \to (x_a, x_b)`.
    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(
        self,
        meta: Callable[[Tensor], Transform],
        mask: BoolTensor,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.meta = meta
        self.idx_a = mask.nonzero().squeeze(-1)
        self.idx_b = (~mask).nonzero().squeeze(-1)

    def split(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return x[..., self.idx_a], x[..., self.idx_b]

    def merge(self, x_a: Tensor, x_b: Tensor, shape: Size) -> Tensor:
        x = x_a.new_empty(shape)
        x[..., self.idx_a] = x_a
        x[..., self.idx_b] = x_b

        return x

    def _call(self, x: Tensor) -> Tensor:
        x_a, x_b = self.split(x)
        y_b = self.meta(x_a)(x_b)

        return self.merge(x_a, y_b, x.shape)

    def _inverse(self, y: Tensor) -> Tensor:
        y_a, y_b = self.split(y)
        x_b = self.meta(y_a).inv(y_b)

        return self.merge(y_a, x_b, y.shape)

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        x_a, x_b = self.split(x)
        _, y_b = self.split(y)

        return self.meta(x_a).log_abs_det_jacobian(x_b, y_b)

    def call_and_ladj(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x_a, x_b = self.split(x)
        y_b, ladj = self.meta(x_a).call_and_ladj(x_b)
        y = self.merge(x_a, y_b, x.shape)

        return y, ladj


class FreeFormJacobianTransform(Transform):
    r"""Creates a free-form Jacobian transformation.

    The transformation is the integration of a system of first-order ordinary
    differential equations

    .. math:: x(t_1) = x_0 + \int_{t_0}^{t_1} f_\phi(t, x(t)) ~ dt .

    References:
        | FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models (Grathwohl et al., 2018)
        | https://arxiv.org/abs/1810.01367

    Arguments:
        f: A system of first-order ODEs :math:`f_\phi`.
        t0: The initial integration time :math:`t_0`.
        t1: The final integration time :math:`t_1`.
        phi: The parameters :math:`\phi` of :math:`f_\phi`.
        atol: The absolute integration tolerance.
        rtol: The relative integration tolerance.
        exact: Whether the exact log-determinant of the Jacobian or an unbiased
            stochastic estimate thereof is calculated.
    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(
        self,
        f: Callable[[Tensor, Tensor], Tensor],
        t0: Union[float, Tensor] = 0.0,
        t1: Union[float, Tensor] = 1.0,
        phi: Iterable[Tensor] = (),
        atol: float = 1e-6,
        rtol: float = 1e-5,
        exact: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.f = f
        self.t0 = t0
        self.t1 = t1
        self.phi = tuple(filter(lambda p: p.requires_grad, phi))
        self.atol = atol
        self.rtol = rtol
        self.exact = exact
        self.trace_scale = 1e-2  # relax jacobian tolerances

    def _call(self, x: Tensor) -> Tensor:
        return odeint(self.f, x, self.t0, self.t1, self.phi, self.atol, self.rtol)

    @property
    def inv(self) -> Transform:
        return FreeFormJacobianTransform(
            f=self.f,
            t0=self.t1,
            t1=self.t0,
            phi=self.phi,
            atol=self.atol,
            rtol=self.rtol,
            exact=self.exact,
        )

    def _inverse(self, y: Tensor) -> Tensor:
        return odeint(self.f, y, self.t1, self.t0, self.phi, self.atol, self.rtol)

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        _, ladj = self.call_and_ladj(x)
        return ladj

    def call_and_ladj(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if self.exact:
            I = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
            I = I.expand(*x.shape, -1).movedim(-1, 0)
        else:
            eps = torch.randn_like(x)

        def f_aug(t: Tensor, x: Tensor, ladj: Tensor) -> Tensor:
            with torch.enable_grad():
                x = x.clone().requires_grad_()
                dx = self.f(t, x)

            if self.exact:
                jacobian = torch.autograd.grad(dx, x, I, create_graph=True, is_grads_batched=True)[0]
                trace = torch.einsum('i...i', jacobian)
            else:
                epsjp = torch.autograd.grad(dx, x, eps, create_graph=True)[0]
                trace = (epsjp * eps).sum(dim=-1)

            return dx, trace * self.trace_scale

        ladj = torch.zeros_like(x[..., 0])
        y, ladj = odeint(f_aug, (x, ladj), self.t0, self.t1, self.phi, self.atol, self.rtol)

        return y, ladj * (1 / self.trace_scale)


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

    def __repr__(self) -> str:
        order = self.order.tolist()

        if len(order) > 10:
            order = order[:5] + [...] + order[-5:]
            order = str(order).replace('Ellipsis', '...')

        return f'{self.__class__.__name__}({order})'

    def _call(self, x: Tensor) -> Tensor:
        return x[..., self.order]

    def _inverse(self, y: Tensor) -> Tensor:
        return y[..., torch.argsort(self.order)]

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.zeros_like(x[..., 0])


class RotationTransform(Transform):
    r"""Creates a rotation transformation :math:`f(x) = R x`.

    .. math:: R = \exp(A - A^T)

    Because :math:`A - A^T` is skew-symmetric, :math:`R` is orthogonal.

    Arguments:
        A: A square matrix :math:`A`, with shape :math:`(*, D, D)`.
    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, A: Tensor, **kwargs):
        super().__init__(**kwargs)

        self.R = torch.linalg.matrix_exp(A - A.mT)

    def _call(self, x: Tensor) -> Tensor:
        return torch.einsum('...ij,...j->...i', self.R, x)

    def _inverse(self, y: Tensor) -> Tensor:
        return torch.einsum('...ij,...i->...j', self.R, y)

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.zeros_like(x[..., 0])


class LULinearTransform(Transform):
    r"""Creates a linear transformation :math:`f(x) = L U x`.

    Arguments:
        LU: A matrix whose lower and upper triangular parts are the non-zero elements
            of :math:`L` and :math:`U`, with shape :math:`(*, D, D)`.
    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, LU: Tensor, **kwargs):
        super().__init__(**kwargs)

        I = torch.eye(LU.shape[-1], dtype=LU.dtype, device=LU.device)

        self.L = torch.tril(LU)
        self.U = torch.triu(LU, diagonal=1) + I

    def _call(self, x: Tensor) -> Tensor:
        return torch.einsum('...ij,...j->...i', self.L @ self.U, x)

    def _inverse(self, y: Tensor) -> Tensor:
        return torch.linalg.solve_triangular(
            self.U,
            torch.linalg.solve_triangular(
                self.L,
                y.unsqueeze(-1),
                upper=False,
                unitriangular=False,
            ),
            upper=True,
            unitriangular=True,
        ).squeeze(-1)

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        diag = torch.diagonal(self.L, dim1=-1, dim2=-2)
        ladj = diag.abs().log().sum(dim=-1)

        return ladj.expand_as(x[..., 0])
