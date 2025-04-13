r"""Autoregressive flows and transformations."""

__all__ = [
    "MAF",
    "MaskedAutoregressiveTransform",
]

import torch

from functools import partial
from math import ceil, prod
from torch import BoolTensor, LongTensor, Size, Tensor
from torch.distributions import Transform
from typing import Callable, Sequence

from .gaussianization import ElementWiseTransform
from ..distributions import DiagNormal
from ..lazy import Flow, LazyTransform, UnconditionalDistribution
from ..nn import MaskedMLP
from ..transforms import AutoregressiveTransform, DependentTransform, MonotonicAffineTransform
from ..utils import broadcast, unpack


class MaskedAutoregressiveTransform(LazyTransform):
    r"""Creates a lazy masked autoregressive transformation.

    See also:
        :class:`zuko.transforms.AutoregressiveTransform`

    References:
        | Masked Autoregressive Flow for Density Estimation (Papamakarios et al., 2017)
        | https://arxiv.org/abs/1705.07057

    Arguments:
        features: The number of features.
        context: The number of context features.
        passes: The number of sequential passes for the inverse transformation. If
            :py:`None`, use the number of features instead, making the transformation
            fully autoregressive. Coupling corresponds to :py:`passes=2`.
        order: A feature ordering. If :py:`None`, use :py:`range(features)` instead.
        adjacency: An adjacency matrix describing the transformation graph. If
            `adjacency` is provided, `order` is ignored and `passes` is replaced by the
            diameter of the graph. Its shape must be either `(features, features)`
            or `(features, features + context)`. If the shape includes context, the rightmost
            `context` columns define connections to the conditioning variables.
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
        tensor([ 1.7428, -1.6483, -0.9920])
        >>> c = torch.randn(4)
        >>> y = t(c)(x)
        >>> t(c).inv(y)
        tensor([ 1.7428, -1.6483, -0.9920], grad_fn=<DivBackward0>)
    """

    def __new__(
        cls,
        features: int = None,
        context: int = 0,
        passes: int = None,
        order: LongTensor = None,
        adjacency: BoolTensor = None,
        *args,
        **kwargs,
    ) -> LazyTransform:
        if features is None or features > 1:
            return super().__new__(cls)
        else:
            return ElementWiseTransform(features, context, *args, **kwargs)

    def __init__(
        self,
        features: int,
        context: int = 0,
        passes: int = None,
        order: LongTensor = None,
        adjacency: BoolTensor = None,
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
        self.register_buffer("order", None)

        if adjacency is None:
            if passes is None:
                passes = features

            if order is None:
                order = torch.arange(features)
            else:
                order = torch.as_tensor(order, dtype=int)

            assert order.ndim == 1, "'order' should be a vector."
            assert order.shape[0] == features, f"'order' should have {features} elements."

            self.passes = min(max(passes, 1), features)
            self.order = torch.div(order, ceil(features / self.passes), rounding_mode="floor")

            adjacency = self.order[:, None] > self.order
            adjacency_context = None
        else:
            adjacency = torch.as_tensor(adjacency, dtype=bool)

            assert adjacency.ndim == 2, "'adjacency' should be a matrix."
            assert adjacency.shape[0] == features, f"'adjacency' should have {features} rows."
            assert adjacency.shape[1] == features or adjacency.shape[1] == features + context, (
                f"'adjacency' should have {features} or {features + context} columns."
            )

            adjacency_context = adjacency[:, features:] if adjacency.shape[1] > features else None
            adjacency = adjacency[:, :features]

            assert adjacency.diag().all(), "'adjacency' should have ones on the diagonal."

            adjacency = adjacency * ~torch.eye(features, dtype=bool)

            self.passes = self._dag_diameter(adjacency)

        if context > 0:
            if adjacency_context is None:
                adjacency_context = torch.ones((features, context), dtype=bool)
            adjacency = torch.cat((adjacency, adjacency_context), dim=1)

        adjacency = torch.repeat_interleave(adjacency, repeats=self.total, dim=0)

        # Hyper network
        self.hyper = MaskedMLP(adjacency, **kwargs)

    @staticmethod
    def _dag_diameter(adjacency: BoolTensor) -> int:
        r"""Returns the diameter of a directed acyclic graph.

        If the graph contains cycles, this function raises an error.

        Credits:
            This code is adapted from :func:`networkx.topological_generations`.

        Arguments:
            adjacency: An adjacency matrix representing a directed graph.

        Returns:
            The diameter of the graph.
        """

        all_generations = []
        indegree = adjacency.sum(dim=1).tolist()
        zero_indegree = [n for n, d in enumerate(indegree) if d == 0]
        while zero_indegree:
            this_generation, zero_indegree = zero_indegree, []
            for node in this_generation:
                for child in adjacency[:, node].nonzero():
                    child = child.item()
                    indegree[child] -= 1
                    if indegree[child] == 0:
                        zero_indegree.append(child)
            all_generations.append(this_generation)

        assert all(d == 0 for d in indegree), "The graph contains cycles."

        return len(all_generations)

    def extra_repr(self) -> str:
        base = self.univariate(*map(torch.randn, self.shapes))

        if self.order is None:
            return "\n".join([
                f"(base): {base}",
                f"(passes): {self.passes}",
            ])
        else:
            order = self.order.tolist()

            if len(order) > 10:
                order = order[:5] + [...] + order[-5:]
                order = str(order).replace("Ellipsis", "...")

            return "\n".join([
                f"(base): {base}",
                f"(order): {order}",
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


class MAF(Flow):
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
          (transform): LazyComposedTransform(
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
          (base): UnconditionalDistribution(DiagNormal(loc: torch.Size([3]), scale: torch.Size([3])))
        )
        >>> c = torch.randn(4)
        >>> x = flow(c).sample()
        >>> x
        tensor([-0.5005, -1.6303,  0.3805])
        >>> flow(c).log_prob(x)
        tensor(-3.7514, grad_fn=<AddBackward0>)
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

        base = UnconditionalDistribution(
            DiagNormal,
            torch.zeros(features),
            torch.ones(features),
            buffer=True,
        )

        super().__init__(transforms, base)
