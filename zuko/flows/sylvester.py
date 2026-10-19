r"""Sylvester flows."""

__all__ = [
    "HSF",
    "OSF",
    "TSF",
    "SylvesterLazyTransform",
]

import math
import torch
import torch.nn as nn

from math import prod
from torch import LongTensor, Tensor
from torch.distributions import Transform
from typing import Literal

from ..distributions import DiagNormal
from ..lazy import Flow, LazyTransform, UnconditionalDistribution
from ..nn import MLP
from ..transforms import SylvesterTransform, TriangularSylvesterTransform
from ..utils import unpack


class SylvesterLazyTransform(LazyTransform):
    r"""Creates a lazy Sylvester transformation.

    The upper triangular matrices :math:`R_1` and :math:`R_2` are parameterized by
    their :math:`M (M + 1) / 2` upper triangular elements. The matrix :math:`Q` with
    orthonormal columns is parameterized according to the :py:`orthogonal` strategy:

    * :py:`'householder'`: the first :math:`M` columns of a product of :math:`K`
      Householder reflections.
    * :py:`'qr'`: the orthonormal factor of the QR decomposition of an unconstrained
      :math:`D \times M` matrix.
    * :py:`'permutation'`: a fixed permutation matrix, which requires :math:`M = D`.

    See also:
        | :class:`zuko.transforms.SylvesterTransform`
        | :class:`zuko.transforms.TriangularSylvesterTransform`

    References:
        | Sylvester Normalizing Flows for Variational Inference (van den Berg et al., 2018)
        | https://arxiv.org/abs/1803.05649

    Arguments:
        features: The number of features.
        context: The number of context features.
        hidden: The number of hidden units :math:`M`. If :py:`None`, :math:`M = D`.
        orthogonal: The parameterization of :math:`Q`. One of
            :py:`['householder', 'qr', 'permutation']`.
        reflections: The number of Householder reflections :math:`K`. If :py:`None`,
            :math:`K = M`. Only used when :py:`orthogonal='householder'`.
        order: The feature permutation, with shape :math:`(D,)`. If :py:`None`, the
            identity is used. Only used when :py:`orthogonal='permutation'`.
        kwargs: Keyword arguments passed to :class:`zuko.nn.MLP`.

    Example:
        >>> t = SylvesterLazyTransform(3, 4)
        >>> t
        SylvesterLazyTransform(
          (orthogonal): householder
          (hyper): MLP(
            (0): Linear(in_features=4, out_features=64, bias=True)
            (1): ReLU()
            (2): Linear(in_features=64, out_features=64, bias=True)
            (3): ReLU()
            (4): Linear(in_features=64, out_features=24, bias=True)
          )
        )
        >>> x = torch.randn(3)
        >>> x
        tensor([ 0.1991,  0.5514, -1.4672])
        >>> c = torch.randn(4)
        >>> y = t(c)(x)
        >>> t(c).inv(y)
        tensor([ 0.1991,  0.5514, -1.4672], grad_fn=<SubBackward0>)
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        hidden: int | None = None,
        orthogonal: Literal["householder", "qr", "permutation"] = "householder",
        reflections: int | None = None,
        order: LongTensor | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if hidden is None:
            hidden = features

        assert hidden <= features, "'hidden' should not be greater than 'features'"

        if orthogonal == "householder":
            if reflections is None:
                reflections = hidden

            shapes = [(reflections, features)]
        elif orthogonal == "qr":
            shapes = [(features, hidden)]
        elif orthogonal == "permutation":
            assert hidden == features, "'hidden' should equal 'features' for permutations"

            shapes = []
        else:
            raise ValueError(f"unknown orthogonal parameterization '{orthogonal}'")

        triu = hidden * (hidden + 1) // 2

        self.hidden = hidden
        self.orthogonal = orthogonal
        self.register_buffer("order", order)

        self.shapes = [*shapes, (triu,), (triu,), (hidden,)]
        self.total = sum(prod(s) for s in self.shapes)

        if context > 0:
            self.hyper = MLP(context, self.total, **kwargs)
        else:
            self.phi = nn.ParameterList(torch.randn(s) / math.sqrt(s[-1]) for s in self.shapes)

    def extra_repr(self) -> str:
        lines = [f"(orthogonal): {self.orthogonal}"]

        if self.order is not None:
            lines.append(f"(order): {self.order.tolist()}")

        return "\n".join(lines)

    def triangular(self, r: Tensor) -> Tensor:
        r"""Scatters the upper triangular elements into an :math:`M \times M` matrix."""

        i, j = torch.triu_indices(self.hidden, self.hidden, device=r.device)

        R = r.new_zeros(*r.shape[:-1], self.hidden, self.hidden)
        R[..., i, j] = r

        return R

    def householder(self, V: Tensor) -> Tensor:
        r"""Returns the first :math:`M` columns of the product of the Householder
        reflections defined by the rows of :math:`V`."""

        V = V / torch.linalg.vector_norm(V, dim=-1, keepdim=True).clamp(min=1e-8)
        Q = torch.eye(V.shape[-1], dtype=V.dtype, device=V.device)[:, : self.hidden]

        for v in torch.unbind(V, dim=-2):
            Q = Q - 2 * v.unsqueeze(-1) * (v.unsqueeze(-2) @ Q)

        return Q

    def forward(self, c: Tensor | None = None) -> Transform:
        if c is None:
            phi = self.phi
        else:
            phi = unpack(self.hyper(c), self.shapes)

        *phi, r1, r2, b = phi
        R1, R2 = self.triangular(r1), self.triangular(r2)

        if self.orthogonal == "householder":
            Q = self.householder(*phi)
        elif self.orthogonal == "qr":
            Q, _ = torch.linalg.qr(*phi)
        else:
            return TriangularSylvesterTransform(R1, R2, b, order=self.order)

        return SylvesterTransform(Q, R1, R2, b)


class TSF(Flow):
    r"""Creates a triangular Sylvester flow (TSF).

    Note:
        Evaluating densities is cheap, but sampling requires inverting Sylvester
        transformations, which involves :math:`M` sequential root-finding problems per
        transformation. When sampling is the bottleneck, as in variational inference,
        the flow can be reversed with :py:`Flow(flow.transform.inv, flow.base)`, which
        makes sampling cheap and density evaluation expensive, as in the original paper.

    See also:
        :class:`SylvesterLazyTransform`

    References:
        | Sylvester Normalizing Flows for Variational Inference (van den Berg et al., 2018)
        | https://arxiv.org/abs/1803.05649

    Arguments:
        features: The number of features.
        context: The number of context features.
        transforms: The number of Sylvester transformations.
        randperm: Whether the features are randomly permuted between transformations
            or not. If :py:`False`, features are in ascending (descending) order for
            even (odd) transformations.
        kwargs: Keyword arguments passed to :class:`SylvesterLazyTransform`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        transforms: int = 3,
        randperm: bool = False,
        **kwargs,
    ) -> None:
        orders = [
            torch.arange(features),
            torch.flipud(torch.arange(features)),
        ]

        transforms = [
            SylvesterLazyTransform(
                features=features,
                context=context,
                orthogonal="permutation",
                order=torch.randperm(features) if randperm else orders[i % 2],
                **kwargs,
            )
            for i in range(transforms)
        ]

        base = UnconditionalDistribution(
            DiagNormal,
            loc=torch.zeros(features),
            scale=torch.ones(features),
            buffer=True,
        )

        super().__init__(transforms, base)


class OSF(Flow):
    r"""Creates an orthogonal Sylvester flow (OSF).

    Note:
        Evaluating densities is cheap, but sampling requires inverting Sylvester
        transformations, which involves :math:`M` sequential root-finding problems per
        transformation. When sampling is the bottleneck, as in variational inference,
        the flow can be reversed with :py:`Flow(flow.transform.inv, flow.base)`, which
        makes sampling cheap and density evaluation expensive, as in the original paper.

    See also:
        :class:`SylvesterLazyTransform`

    References:
        | Sylvester Normalizing Flows for Variational Inference (van den Berg et al., 2018)
        | https://arxiv.org/abs/1803.05649

    Arguments:
        features: The number of features.
        context: The number of context features.
        transforms: The number of Sylvester transformations.
        kwargs: Keyword arguments passed to :class:`SylvesterLazyTransform`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        transforms: int = 3,
        **kwargs,
    ) -> None:
        transforms = [
            SylvesterLazyTransform(
                features=features,
                context=context,
                orthogonal="qr",
                **kwargs,
            )
            for _ in range(transforms)
        ]

        base = UnconditionalDistribution(
            DiagNormal,
            loc=torch.zeros(features),
            scale=torch.ones(features),
            buffer=True,
        )

        super().__init__(transforms, base)


class HSF(Flow):
    r"""Creates a Householder Sylvester flow (HSF).

    Note:
        Evaluating densities is cheap, but sampling requires inverting Sylvester
        transformations, which involves :math:`M` sequential root-finding problems per
        transformation. When sampling is the bottleneck, as in variational inference,
        the flow can be reversed with :py:`Flow(flow.transform.inv, flow.base)`, which
        makes sampling cheap and density evaluation expensive, as in the original paper.

    See also:
        :class:`SylvesterLazyTransform`

    References:
        | Sylvester Normalizing Flows for Variational Inference (van den Berg et al., 2018)
        | https://arxiv.org/abs/1803.05649

    Arguments:
        features: The number of features.
        context: The number of context features.
        transforms: The number of Sylvester transformations.
        kwargs: Keyword arguments passed to :class:`SylvesterLazyTransform`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        transforms: int = 3,
        **kwargs,
    ) -> None:
        transforms = [
            SylvesterLazyTransform(
                features=features,
                context=context,
                orthogonal="householder",
                **kwargs,
            )
            for _ in range(transforms)
        ]

        base = UnconditionalDistribution(
            DiagNormal,
            loc=torch.zeros(features),
            scale=torch.ones(features),
            buffer=True,
        )

        super().__init__(transforms, base)
