r"""Autoregressive flows and transformations."""

__all__ = [
    'MAF',
    'MaskedAutoregressiveTransform',
]

import torch

from functools import partial
from math import ceil, prod
from torch import BoolTensor, LongTensor, Size, Tensor
from torch.distributions import Transform
from typing import Callable, Sequence

# isort: split
from .core import Flow, LazyTransform, UnconditionalDistribution
from .gaussianization import ElementWiseTransform
from ..distributions import DiagNormal
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
        order: The feature ordering. If :py:`None`, use :py:`range(features)` instead.
        adjacency: The adjacency matrix describing the factorization of the
            joint distribution. If different from :py:`None`, then `order` must be
            :py:`None`. If `passes` is :py:`None`, then `passes` is set to the
            diameter of the matrix described by `adjacency`.
        univariate: The univariate transformation constructor.
        shapes: The shapes of the univariate transformation parameters.
        kwargs: Keyword arguments passed to :class:`zuko.nn.MaskedMLP`.

    Example:
        >>> t = MaskedAutoregressiveTransform(3, 4)
        >>> t
        MaskedAutoregressiveTransform(
          (base): MonotonicAffineTransform()
          (order): [0, 1, 2]
          (adjacency): None
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
        self.register_buffer('order', None)

        assert (order is None) or (
            adjacency is None
        ), "Parameters `order` and `adjacency_matrix` are mutually exclusive."

        if adjacency is None:
            if passes is None:
                passes = features
            self.passes = min(max(passes, 1), features)

            if order is None:
                order = torch.arange(features)
            else:
                order = torch.as_tensor(order)

            self.order = torch.div(order, ceil(features / self.passes), rounding_mode='floor')

            in_order = torch.cat((self.order, torch.full((context,), -1)))
            out_order = torch.repeat_interleave(self.order, self.total)
            adjacency = out_order[:, None] > in_order
        else:
            diameter = self._check_adjacency(adjacency)
            if passes is None:
                self.passes = diameter

            adjacency = torch.cat(
                (adjacency, torch.ones((adjacency.shape[0], context), dtype=bool)), dim=1
            )
            adjacency = torch.repeat_interleave(adjacency.bool(), repeats=self.total, dim=0)

        # Hyper network
        self.hyper = MaskedMLP(adjacency, **kwargs)

    def _check_adjacency(self, adjacency: BoolTensor) -> int:
        r"""Checks that adjacency is valid (squared tensor, zeroed diagonal, and acyclic)
        Args:
            adjacency: The adjacency matrix.

        Returns:
            The diameter of the adjacency matrix (which describes the number of passes).
            Based on the code for computing the topological generations in networkx
            # https://networkx.org/documentation/stable/_modules/networkx/algorithms/dag.html#is_directed_acyclic_graph
        """
        assert (len(adjacency.size()) == 2) and (
            adjacency.size(0) == adjacency.size(1)
        ), "`adjacency` should be a 2-dimensional squared tensor (a matrix)."

        assert adjacency.diag().all(), "The diagonal of `adjacency` should be all ones."
        adjacency.mul_(  # Remove the diagonal
            ~torch.eye(adjacency.size(0), dtype=torch.bool, device=adjacency.device)
        )

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

        return len(all_generations)  # Graph diameter

    def extra_repr(self) -> str:
        base = self.univariate(*map(torch.randn, self.shapes))
        order = self.order

        if self.order is None:
            adjacency = 'Set!'  # TODO I don't want to show the entire matrix.
        else:
            adjacency = None
            order = order.tolist()
            if len(order) > 10:
                order = order[:5] + [...] + order[-5:]
                order = str(order).replace('Ellipsis', '...')

        return '\n'.join([
            f'(base): {base}',
            f'(order): {order}',
            f'(adjacency): {adjacency}',
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
              (adjacency): None
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
              (adjacency): None
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
              (adjacency): None
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
        orders = (
            torch.arange(features),
            torch.flipud(torch.arange(features)),
        )

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
