r"""Parameterizable probability distributions."""

__all__ = [
    "BoxUniform",
    "DiagNormal",
    "GeneralizedNormal",
    "Joint",
    "Maximum",
    "Minimum",
    "Mixture",
    "NormalizingFlow",
    "Sort",
    "TopK",
    "TransformedUniform",
    "Truncated",
]

import math
import torch

from textwrap import indent
from torch import Size, Tensor
from torch.distributions import *  # noqa: F403
from torch.distributions import (
    Categorical,
    Distribution,
    Independent,
    Normal,
    Transform,
    Uniform,
    constraints,
)
from torch.distributions.utils import _sum_rightmost
from typing import Tuple

Distribution._validate_args = False
Distribution.arg_constraints = {}


class NormalizingFlow(Distribution):
    r"""Creates a normalizing flow for a random variable :math:`X` towards a base
    distribution :math:`p(Z)` through a transformation :math:`f`.

    The density of a realization :math:`x` is given by the change of variables

    .. math:: p(X = x) = p(Z = f(x)) \left| \det \frac{\partial f(x)}{\partial x} \right| .

    To sample from :math:`p(X)`, realizations :math:`z \sim p(Z)` are mapped through
    the inverse transformation :math:`g = f^{-1}`.

    References:
        | A Family of Non-parametric Density Estimation Algorithms (Tabak et al., 2013)
        | https://onlinelibrary.wiley.com/doi/abs/10.1002/cpa.21423

        | Variational Inference with Normalizing Flows (Rezende et al., 2015)
        | https://arxiv.org/abs/1505.05770

        | Normalizing Flows for Probabilistic Modeling and Inference (Papamakarios et al., 2021)
        | https://arxiv.org/abs/1912.02762

    Arguments:
        transform: A transformation :math:`f`.
        base: A base distribution :math:`p(Z)`.

    Example:
        >>> d = NormalizingFlow(ExpTransform(), Gamma(2.0, 1.0))
        >>> d.sample()
        tensor(1.5157)
    """

    has_rsample = True

    def __init__(
        self,
        transform: Transform,
        base: Distribution,
    ):
        super().__init__()

        reinterpreted = transform.codomain.event_dim - len(base.event_shape)

        if reinterpreted > 0:
            base = Independent(base, reinterpreted)

        self.transform = transform
        self.base = base
        self.reinterpreted = max(-reinterpreted, 0)

    def __repr__(self) -> str:
        lines = [
            f"(transform): {self.transform}",
            f"(base): {self.base}",
        ]
        lines = indent("\n".join(lines), "  ")

        return self.__class__.__name__ + "(\n" + lines + "\n)"

    @property
    def batch_shape(self) -> Size:
        return self.base.batch_shape

    @property
    def event_shape(self) -> Size:
        return self.transform.inverse_shape(self.base.event_shape)

    def expand(self, batch_shape: Size, new: Distribution = None) -> Distribution:
        new = self._get_checked_instance(NormalizingFlow, new)
        new.transform = self.transform
        new.base = self.base.expand(batch_shape)
        new.reinterpreted = self.reinterpreted

        Distribution.__init__(new, validate_args=False)

        return new

    def log_prob(self, x: Tensor) -> Tensor:
        z, ladj = self.transform.call_and_ladj(x)
        ladj = _sum_rightmost(ladj, self.reinterpreted)

        return self.base.log_prob(z) + ladj

    def rsample(self, shape: Size = ()) -> Tensor:
        if self.base.has_rsample:
            z = self.base.rsample(shape)
        else:
            z = self.base.sample(shape)

        return self.transform.inv(z)

    def rsample_and_log_prob(self, shape: Size = ()) -> Tuple[Tensor, Tensor]:
        if self.base.has_rsample:
            z = self.base.rsample(shape)
        else:
            z = self.base.sample(shape)

        x, ladj = self.transform.inv.call_and_ladj(z)
        ladj = _sum_rightmost(ladj, self.reinterpreted)

        return x, self.base.log_prob(z) - ladj


class Joint(Distribution):
    r"""Creates a distribution for a multivariate random variable :math:`X` which
    is the concatenation of :math:`n` independent random variables :math:`Z_i`.

    .. math:: p(X = x) = \prod_i p(Z_i = x_i)

    Arguments:
        marginals: A list of distributions :math:`p(Z_i)`.

    Example:
        >>> d = Joint(Uniform(0.0, 1.0), Normal(0.0, 1.0))
        >>> d.event_shape
        torch.Size([2])
        >>> d.sample()
        tensor([0.4963, 0.2072])
    """

    has_rsample = True

    def __init__(self, *marginals: Distribution):
        super().__init__()

        batch_shape = torch.broadcast_shapes(*(m.batch_shape for m in marginals))

        self.marginals = [m.expand(batch_shape) for m in marginals]

    def __repr__(self) -> str:
        lines = map(repr, self.marginals)
        lines = indent("\n".join(lines), "  ")

        return self.__class__.__name__ + "(\n" + lines + "\n)"

    @property
    def batch_shape(self) -> Size:
        return self.marginals[0].batch_shape

    @property
    def event_shape(self) -> Size:
        return Size([sum(m.event_shape.numel() for m in self.marginals)])

    def expand(self, batch_shape: Size, new: Distribution = None) -> Distribution:
        new = self._get_checked_instance(Joint, new)
        new.marginals = [m.expand(batch_shape) for m in self.marginals]

        Distribution.__init__(new, validate_args=False)

        return new

    def log_prob(self, x: Tensor) -> Tensor:
        shape = x.shape[:-1]
        i, lp = 0, 0

        for m in self.marginals:
            j = i + m.event_shape.numel()
            z = x[..., i:j].reshape(shape + m.event_shape)
            lp = lp + m.log_prob(z)
            i = j

        return lp

    def rsample(self, shape: Size = ()) -> Tensor:
        x = []

        for m in self.marginals:
            if m.has_rsample:
                z = m.rsample(shape)
            else:
                z = m.sample(shape)

            z = z.reshape(shape + m.batch_shape + (-1,))
            x.append(z)

        return torch.cat(x, dim=-1)


class Mixture(Distribution):
    r"""Creates a mixture of distributions for a random variable :math:`X`.

    .. math:: p(X = x) = \frac{1}{\sum_i w_i} \sum_i w_i \, p(Z_i = x)

    Wikipedia:
        https://wikipedia.org/wiki/Mixture_model

    Arguments:
        base: A batch of base distributions :math:`p(Z_i)`.
        logits: The unnormalized log-weights :math:`\log w_i`.

    Example:
        >>> d = Mixture(Normal(torch.randn(2), torch.ones(2)), torch.randn(2))
        >>> d.event_shape
        torch.Size([])
        >>> d.sample()
        tensor(-1.6920)
    """

    has_rsample = False

    def __init__(self, base: Distribution, logits: Tensor):
        super().__init__()

        assert base.batch_shape[-1] == logits.shape[-1]
        assert len(base.batch_shape) >= len(logits.shape)

        self.base = base
        self.logits = logits

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.base})"

    @property
    def batch_shape(self) -> Size:
        return self.base.batch_shape[:-1]

    @property
    def event_shape(self) -> Size:
        return self.base.event_shape

    def expand(self, batch_shape: Size, new: Distribution = None) -> Distribution:
        new = self._get_checked_instance(Mixture, new)
        new.base = self.base.expand(batch_shape + self.base.batch_shape[-1:])
        new.logits = self.logits

        Distribution.__init__(new, validate_args=False)

        return new

    def log_prob(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(dim=-(len(self.event_shape) + 1))

        log_w = torch.log_softmax(self.logits, dim=-1)
        log_p = self.base.log_prob(x)

        return torch.logsumexp(log_w + log_p, dim=-1)

    def sample(self, shape: Size = ()) -> Tensor:
        with torch.no_grad():
            x = self.base.sample(shape)
            i = Categorical(logits=self.logits).expand(self.batch_shape).sample(shape)

            if i.dim() > 0:
                x = x.flatten(0, i.dim() - 1)
                x = x[torch.arange(i.numel(), device=i.device), i.flatten()]

                return x.reshape(shape + self.batch_shape + self.event_shape)
            else:
                return x[i]


class GeneralizedNormal(Distribution):
    r"""Creates a generalized normal distribution.

    .. math:: p(X = x) = \frac{\beta}{2 \Gamma(1 / \beta)} \exp(-|x|^\beta)

    Wikipedia:
        https://wikipedia.org/wiki/Generalized_normal_distribution

    Arguments:
        beta: The shape parameter :math:`\beta`.

    Example:
        >>> d = GeneralizedNormal(2.0)
        >>> d.sample()
        tensor(-0.0281)
    """

    arg_constraints = {"beta": constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, beta: Tensor):
        super().__init__()

        self.beta = torch.as_tensor(beta)

    @property
    def batch_shape(self) -> Size:
        return self.beta.shape

    def expand(self, batch_shape: Size, new: Distribution = None) -> Distribution:
        new = self._get_checked_instance(GeneralizedNormal, new)
        new.beta = self.beta.expand(batch_shape)

        Distribution.__init__(new, validate_args=False)

        return new

    def log_prob(self, x: Tensor) -> Tensor:
        return torch.log(self.beta / 2) - torch.lgamma(1 / self.beta) - abs(x) ** self.beta

    def rsample(self, shape: Size = ()) -> Tensor:
        beta = self.beta.expand(shape + self.beta.shape)
        x = torch._standard_gamma(1 / beta) ** (1 / beta)
        x = x * torch.sign(2 * torch.rand_like(x) - 1)
        return x


class DiagNormal(Independent):
    r"""Creates a multivariate normal distribution parametrized by the variables
    mean :math:`\mu` and standard deviation :math:`\sigma`, but assumes no
    correlation between the variables.

    Arguments:
        loc: The mean :math:`\mu` of the variables.
        scale: The standard deviation :math:`\sigma` of the variables.
        ndims: The number of batch dimensions to interpret as event dimensions.

    Example:
        >>> d = DiagNormal(torch.zeros(3), torch.ones(3))
        >>> d.event_shape
        torch.Size([3])
        >>> d.sample()
        tensor([ 1.5410, -0.2934, -2.1788])
    """

    def __init__(self, loc: Tensor, scale: Tensor, ndims: int = 1):
        super().__init__(Normal(torch.as_tensor(loc), torch.as_tensor(scale)), ndims)

    def __repr__(self) -> str:
        return "Diag" + repr(self.base_dist)

    def expand(self, batch_shape: Size, new: Distribution = None) -> Distribution:
        new = self._get_checked_instance(DiagNormal, new)
        return super().expand(batch_shape, new)


class BoxUniform(Independent):
    r"""Creates a distribution for a multivariate random variable :math:`X`
    distributed uniformly over an hypercube domain. Formally,

    .. math:: l_i \leq X_i < u_i ,

    where :math:`l_i` and :math:`u_i` are respectively the lower and upper bounds
    of the domain in the :math:`i`-th dimension.

    Arguments:
        lower: The lower bounds (inclusive).
        upper: The upper bounds (exclusive).
        ndims: The number of batch dimensions to interpret as event dimensions.

    Example:
        >>> d = BoxUniform(-torch.ones(3), torch.ones(3))
        >>> d.event_shape
        torch.Size([3])
        >>> d.sample()
        tensor([-0.0075,  0.5364, -0.8230])
    """

    def __init__(self, lower: Tensor, upper: Tensor, ndims: int = 1):
        super().__init__(Uniform(torch.as_tensor(lower), torch.as_tensor(upper)), ndims)

    def __repr__(self) -> str:
        return "Box" + repr(self.base_dist)

    def expand(self, batch_shape: Size, new: Distribution = None) -> Distribution:
        new = self._get_checked_instance(BoxUniform, new)
        return super().expand(batch_shape, new)


class TransformedUniform(NormalizingFlow):
    r"""Creates a distribution for a random variable :math:`X`, whose
    transformation :math:`f(X)` is uniformly distributed over the interval
    :math:`[f(l), f(u)]`.

    .. math:: p(X = x) = \frac{1}{f(u) - f(l)}
        \begin{cases}
            f'(x) & \text{if } f(l) \leq f(x) < f(u) \\
            0 & \text{otherwise}
        \end{cases}

    Arguments:
        f: A transformation :math:`f`, monotonically increasing over :math:`[l, u]`.
        lower: A lower bound :math:`l` (inclusive).
        upper: An upper bound :math:`u` (exclusive).

    Example:
        >>> d = TransformedUniform(ExpTransform(), -1.0, 1.0)
        >>> d.sample()
        tensor(0.4281)
    """

    def __init__(self, f: Transform, lower: Tensor, upper: Tensor):
        super().__init__(f, Uniform(*map(f, map(torch.as_tensor, (lower, upper)))))

    def expand(self, batch_shape: Size, new: Distribution = None) -> Distribution:
        new = self._get_checked_instance(TransformedUniform, new)
        return super().expand(batch_shape, new)


class Truncated(Distribution):
    r"""Truncates a base distribution :math:`p(X)` between a lower bound
    :math:`l` and an upper bound :math:`u`.

    .. math:: p(X = x | l \leq X < u) = \frac{1}{P(X \leq u) - P(X \leq l)}
        \begin{cases}
            p(X = x) & \text{if } l \leq x < u \\
            0 & \text{otherwise}
        \end{cases}

    Arguments:
        base: A base distribution :math:`p(X)`.
        lower: A lower bound :math:`l` (inclusive).
        upper: An upper bound :math:`u` (exclusive).

    Example:
        >>> d = Truncated(Normal(0.0, 1.0), 1.0, 2.0)
        >>> d.sample()
        tensor(1.3333)
    """

    has_rsample = True

    def __init__(
        self,
        base: Distribution,
        lower: Tensor = float("-inf"),
        upper: Tensor = float("+inf"),
    ):
        super().__init__()

        assert not base.event_shape, "'base' has to be univariate"

        self.base = base
        self.uniform = Uniform(base.cdf(lower), base.cdf(upper))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.base})"

    @property
    def batch_shape(self) -> Size:
        return self.base.batch_shape

    def expand(self, batch_shape: Size, new: Distribution = None) -> Distribution:
        new = self._get_checked_instance(Truncated, new)
        new.base = self.base.expand(batch_shape)
        new.uniform = self.uniform.expand(batch_shape)

        Distribution.__init__(new, validate_args=False)

        return new

    def cdf(self, x: Tensor) -> Tensor:
        return self.uniform.cdf(self.base.cdf(x))

    def log_prob(self, x: Tensor) -> Tensor:
        return self.uniform.log_prob(self.base.cdf(x)) + self.base.log_prob(x)

    def rsample(self, shape: Size = ()) -> Tensor:
        return self.base.icdf((1 - 2e-6) * self.uniform.rsample(shape) + 1e-6)


class Sort(Distribution):
    r"""Creates a distribution for a :math:`n`-d random variable :math:`X`, whose elements
    :math:`X_i` are :math:`n` draws from a base distribution :math:`p(Z)`, ordered
    such that :math:`X_i \leq X_{i + 1}`.

    .. math:: p(X = x) = \begin{cases}
            n! \, \prod_{i = 1}^n p(Z = x_i) & \text{if $x$ is ordered} \\
            0 & \text{otherwise}
        \end{cases}

    Arguments:
        base: A base distribution :math:`p(Z)`.
        n: The number of draws :math:`n`.
        descending: Whether the elements are sorted in descending order or not.

    Example:
        >>> d = Sort(Normal(0.0, 1.0), 3)
        >>> d.event_shape
        torch.Size([3])
        >>> d.sample()
        tensor([-2.1788, -0.2934,  1.5410])
    """

    def __init__(
        self,
        base: Distribution,
        n: int = 2,
        descending: bool = False,
    ):
        super().__init__()

        assert not base.event_shape, "'base' has to be univariate"

        self.base = base
        self.n = n
        self.descending = descending
        self.log_fact = math.log(math.factorial(n))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.base}, {self.n})"

    @property
    def batch_shape(self) -> Size:
        return self.base.batch_shape

    @property
    def event_shape(self) -> Size:
        return Size([self.n])

    def expand(self, batch_shape: Size, new: Distribution = None) -> Distribution:
        new = self._get_checked_instance(Sort, new)
        new.base = self.base.expand(batch_shape)
        new.n = self.n
        new.descending = self.descending
        new.log_fact = self.log_fact

        Distribution.__init__(new, validate_args=False)

        return new

    def log_prob(self, x: Tensor) -> Tensor:
        x = torch.movedim(x, -1, 0)

        if self.descending:
            ordered = x[:-1] >= x[1:]
        else:
            ordered = x[:-1] <= x[1:]

        ordered = ordered.all(dim=0)

        return ordered.log() + self.log_fact + self.base.log_prob(x).sum(dim=0)

    def sample(self, shape: Size = ()) -> Tensor:
        x = torch.movedim(self.base.sample((self.n, *shape)), 0, -1)
        x = torch.sort(x, dim=-1, descending=self.descending).values

        return x


class TopK(Sort):
    r"""Creates a distribution for a :math:`k`-d random variable :math:`X`, whose elements
    :math:`X_i` are the top :math:`k` among :math:`n` draws from a base distribution
    :math:`p(Z)`, ordered such that :math:`X_i \leq X_{i + 1}`.

    .. math:: p(X = x) = \begin{cases}
            \frac{n!}{(n - k)!} \, \prod_{i = 1}^k p(Z = x_i)
                \, P(Z \geq x_k)^{n - k} & \text{if $x$ is ordered} \\
            0 & \text{otherwise}
        \end{cases}

    Arguments:
        base: A base distribution :math:`p(Z)`.
        k: The number of selected elements :math:`k`.
        n: The number of draws :math:`n`.
        kwargs: Keyword arguments passed to :class:`Sort`.

    Example:
        >>> d = TopK(Normal(0.0, 1.0), 2, 3)
        >>> d.event_shape
        torch.Size([2])
        >>> d.sample()
        tensor([-2.1788, -0.2934])
    """

    def __init__(
        self,
        base: Distribution,
        k: int = 1,
        n: int = 2,
        **kwargs,
    ):
        super().__init__(base, n, **kwargs)

        assert 1 <= k < n, "k has to be in [1, n)"

        self.k = k
        self.log_fact = self.log_fact - math.log(math.factorial(n - k))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.base}, {self.k}, {self.n})"

    @property
    def event_shape(self) -> Size:
        return Size([self.k])

    def expand(self, batch_shape: Size, new: Distribution = None) -> Distribution:
        new = self._get_checked_instance(TopK, new)
        new.k = self.k
        return super().expand(batch_shape, new)

    def log_prob(self, x: Tensor) -> Tensor:
        cdf = self.base.cdf(x[..., -1])

        if not self.descending:
            cdf = 1 - cdf

        return (self.n - self.k) * cdf.log() + super().log_prob(x)

    def sample(self, shape: Size = ()) -> Tensor:
        return super().sample(shape)[..., : self.k]


class Minimum(TopK):
    r"""Creates a distribution for a random variable :math:`X`, which is the
    minimum among :math:`n` draws from a base distribution :math:`p(Z)`.

    .. math:: p(X = x) = n \, p(Z = x) \, P(Z \geq x)^{n - 1}

    Arguments:
        base: A base distribution :math:`p(Z)`.
        n: The number of draws :math:`n`.

    Example:
        >>> d = Minimum(Normal(0.0, 1.0), 3)
        >>> d.event_shape
        torch.Size([])
        >>> d.sample()
        tensor(-2.1788)
    """

    def __init__(self, base: Distribution, n: int = 2):
        super().__init__(base, 1, n)

        self.descending = False

    def __repr__(self) -> str:
        return Sort.__repr__(self)

    @property
    def event_shape(self) -> Size:
        return Size([])

    def expand(self, batch_shape: Size, new: Distribution = None) -> Distribution:
        new = self._get_checked_instance(Minimum, new)
        return super().expand(batch_shape, new)

    def log_prob(self, x: Tensor) -> Tensor:
        return super().log_prob(x.unsqueeze(dim=-1))

    def sample(self, shape: Size = ()) -> Tensor:
        return super().sample(shape).squeeze(dim=-1)


class Maximum(Minimum):
    r"""Creates a distribution for a random variable :math:`X`, which is the
    maximum among :math:`n` draws from a base distribution :math:`p(Z)`.

    .. math:: p(X = x) = n \, p(Z = x) \, P(Z \leq x)^{n - 1}

    Arguments:
        base: A base distribution :math:`p(Z)`.
        n: The number of draws :math:`n`.

    Example:
        >>> d = Maximum(Normal(0.0, 1.0), 3)
        >>> d.event_shape
        torch.Size([])
        >>> d.sample()
        tensor(1.5410)
    """

    def __init__(self, base: Distribution, n: int = 2):
        super().__init__(base, n)

        self.descending = True

    def expand(self, batch_shape: Size, new: Distribution = None) -> Distribution:
        new = self._get_checked_instance(Maximum, new)
        return super().expand(batch_shape, new)
