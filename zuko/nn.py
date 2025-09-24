r"""Neural networks, layers and modules."""

__all__ = ["MLP", "Linear", "MaskedMLP", "MonotonicMLP"]

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import BoolTensor, Tensor
from typing import Callable, Iterable, Sequence, Union


def linear(x: Tensor, W: Tensor, b: Tensor = None) -> Tensor:
    if W.dim() == 2:
        return F.linear(x, W, b)
    else:
        x = torch.einsum("...ij,...j->...i", W, x)

    if b is None:
        return x
    else:
        return x + b


class LayerNorm(nn.Module):
    r"""Creates a normalization layer that standardizes features along a dimension.

    .. math:: y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}}

    References:
       Layer Normalization (Lei Ba et al., 2016)
       https://arxiv.org/abs/1607.06450

    Arguments:
        dim: The dimension(s) to standardize.
        eps: A numerical stability term.
    """

    def __init__(self, dim: Union[int, Iterable[int]] = -1, eps: float = 1e-5):
        super().__init__()

        self.dim = dim if isinstance(dim, int) else tuple(dim)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        variance, mean = torch.var_mean(x, unbiased=True, dim=self.dim, keepdim=True)

        return (x - mean) / (variance + self.eps).sqrt()


class Linear(nn.Module):
    r"""Creates a linear layer.

    .. math:: y = x W^T + b

    If the :py:`stack` argument is provided, the module creates a stack of
    independent linear operators that are applied to a stack of input vectors.

    Arguments:
        in_features: The number of input features :math:`C`.
        out_features: The number of output features :math:`C'`.
        bias: Whether the layer learns an additive bias :math:`b` or not.
        stack: The number of stacked operators :math:`S`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        stack: int = None,
    ):
        super().__init__()

        shape = () if stack is None else (stack,)

        self.weight = nn.Parameter(torch.empty(*shape, out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(*shape, out_features))
        else:
            self.bias = None

        self.reset_parameters()

        self.in_features = in_features
        self.out_features = out_features

    def reset_parameters(self):
        bound = 1 / self.weight.shape[-1] ** 0.5

        nn.init.uniform_(self.weight, -bound, bound)

        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        if self.weight.dim() == 2:
            stack, fout, fin = (None, *self.weight.shape)
        else:
            stack, fout, fin = self.weight.shape

        bias = self.bias is not None

        if stack is None:
            return f"in_features={fin}, out_features={fout}, bias={bias}"
        else:
            return f"in_features={fin}, out_features={fout}, bias={bias}, stack={stack}"

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor :math:`x`, with shape :math:`(*, C)` or :math:`(*, S, C)`.

        Returns:
            The output tensor :math:`y`, with shape or :math:`(*, C')` or :math:`(*, S, C')`.
        """

        return linear(x, self.weight, self.bias)


class MLP(nn.Sequential):
    r"""Creates a multi-layer perceptron (MLP).

    Also known as fully connected feedforward network, an MLP is a sequence of
    non-linear parametric functions

    .. math:: h_{i + 1} = a_{i + 1}(h_i W_{i + 1}^T + b_{i + 1}),

    over feature vectors :math:`h_i`, with the input and output feature vectors
    :math:`x = h_0` and :math:`y = h_L`, respectively. The non-linear functions
    :math:`a_i` are called activation functions. The trainable parameters of an MLP
    are its weights and biases :math:`\phi = \{W_i, b_i | i = 1, \dots, L\}`.

    Wikipedia:
        https://wikipedia.org/wiki/Feedforward_neural_network

    Arguments:
        in_features: The number of input features.
        out_features: The number of output features.
        hidden_features: The numbers of hidden features.
        activation: The activation function constructor. If :py:`None`, use
            :class:`torch.nn.ReLU` instead.
        normalize: Whether features are normalized between layers or not.
        kwargs: Keyword arguments passed to :class:`Linear`.

    Example:
        >>> net = MLP(64, 1, [32, 16], activation=nn.ELU)
        >>> net
        MLP(
          (0): Linear(in_features=64, out_features=32, bias=True)
          (1): ELU(alpha=1.0)
          (2): Linear(in_features=32, out_features=16, bias=True)
          (3): ELU(alpha=1.0)
          (4): Linear(in_features=16, out_features=1, bias=True)
        )
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Sequence[int] = (64, 64),
        activation: Callable[[], nn.Module] = None,
        normalize: bool = False,
        **kwargs,
    ):
        if activation is None:
            activation = nn.ReLU

        normalization = LayerNorm if normalize else lambda: None

        layers = []

        for before, after in zip(
            (in_features, *hidden_features),
            (*hidden_features, out_features),
        ):
            layers.extend([
                Linear(before, after, **kwargs),
                activation(),
                normalization(),
            ])

        layers = layers[:-2]
        layers = filter(lambda layer: layer is not None, layers)

        super().__init__(*layers)

        self.in_features = in_features
        self.out_features = out_features


class Residual(nn.Sequential):
    r"""Creates a residual block."""

    def forward(self, x: Tensor) -> Tensor:
        return x + super().forward(x)


class MaskedLinear(nn.Linear):
    r"""Creates a masked linear layer.

    .. math:: y = x (W \odot A)^T + b

    Arguments:
        adjacency: The adjacency matrix :math:`A \in \{0, 1\}^{M \times N}`.
        kwargs: Keyword arguments passed to :class:`torch.nn.Linear`.
    """

    def __init__(self, adjacency: BoolTensor, **kwargs):
        super().__init__(*reversed(adjacency.shape), **kwargs)

        self.register_buffer("mask", adjacency)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.mask * self.weight, self.bias)


class MaskedMLP(nn.Sequential):
    r"""Creates a masked multi-layer perceptron (MaskedMLP).

    The resulting MLP is a transformation :math:`y = f(x)` whose Jacobian entries
    :math:`\frac{\partial y_i}{\partial x_j}` are null if :math:`A_{ij} = 0`.

    Arguments:
        adjacency: The adjacency matrix :math:`A \in \{0, 1\}^{M \times N}`.
        hidden_features: The numbers of hidden features.
        activation: The activation function constructor. If :py:`None`, use
            :class:`torch.nn.ReLU` instead.
        residual: Whether to use residual blocks or not.

    Example:
        >>> adjacency = torch.randn(4, 3) < 0
        >>> adjacency
        tensor([[False,  True,  True],
                [False,  True,  True],
                [False, False,  True],
                [ True,  True, False]])
        >>> net = MaskedMLP(adjacency, [16, 32], activation=nn.ELU)
        >>> net
        MaskedMLP(
          (0): MaskedLinear(in_features=3, out_features=16, bias=True)
          (1): ELU(alpha=1.0)
          (2): MaskedLinear(in_features=16, out_features=32, bias=True)
          (3): ELU(alpha=1.0)
          (4): MaskedLinear(in_features=32, out_features=4, bias=True)
        )
        >>> x = torch.randn(3)
        >>> torch.autograd.functional.jacobian(net, x)
        tensor([[ 0.0000, -0.0065,  0.1158],
                [ 0.0000, -0.0089,  0.0072],
                [ 0.0000,  0.0000,  0.0089],
                [-0.0146, -0.0128,  0.0000]])
    """

    def __init__(
        self,
        adjacency: BoolTensor,
        hidden_features: Sequence[int] = (64, 64),
        activation: Callable[[], nn.Module] = None,
        residual: bool = False,
    ):
        out_features, in_features = adjacency.shape

        if activation is None:
            activation = nn.ReLU

        # Merge outputs with the same dependencies
        adjacency, inverse = torch.unique(adjacency, dim=0, return_inverse=True)

        # P_ij = 1 if A_ik = 1 for all k such that A_jk = 1
        precedence = adjacency.double() @ adjacency.double().t() == adjacency.sum(dim=-1)

        # Layers
        layers = []

        for i, features in enumerate((*hidden_features, out_features)):
            if i > 0:
                mask = precedence[:, indices]  # noqa: F821
            else:
                mask = adjacency

            if (~mask).all():
                raise ValueError("The adjacency matrix leads to a null Jacobian.")

            if i < len(hidden_features):
                reachable = mask.sum(dim=-1).nonzero().squeeze(dim=-1)
                indices = reachable[torch.arange(features) % len(reachable)]
                mask = mask[indices]
            else:
                mask = mask[inverse]

            layers.append(MaskedLinear(adjacency=mask))

            if residual:
                if 0 < i < len(hidden_features) and mask.shape[0] == mask.shape[1]:
                    layers.pop()

                mask = precedence[indices, :][:, indices]

                layers.append(
                    Residual(
                        MaskedLinear(adjacency=mask),
                        activation(),
                        MaskedLinear(adjacency=mask),
                    )
                )
            else:
                layers.append(activation())

        layers.pop()

        super().__init__(*layers)

        self.in_features = in_features
        self.out_features = out_features


class MonotonicLinear(Linear):
    r"""Creates a monotonic linear layer.

    .. math:: y = x |W|^T + b

    Arguments:
        args: Positional arguments passed to :class:`Linear`.
        kwargs: Keyword arguments passed to :class:`Linear`.
    """

    def forward(self, x: Tensor) -> Tensor:
        return linear(x, self.weight.abs(), self.bias)


class TwoWayELU(nn.ELU):
    r"""Creates a layer that splits the input into two groups and applies
    :math:`\text{ELU}(x)` to the first and :math:`-\text{ELU}(-x)` to the second.

    Arguments:
        args: Positional arguments passed to :class:`torch.nn.ELU`.
        kwargs: Keyword arguments passed to :class:`torch.nn.ELU`.
    """

    def forward(self, x: Tensor) -> Tensor:
        x0, x1 = torch.chunk(x, 2, dim=-1)

        return torch.cat(
            (
                super().forward(x0),
                -super().forward(-x1),
            ),
            dim=-1,
        )


class MonotonicMLP(MLP):
    r"""Creates a monotonic multi-layer perceptron (MonotonicMLP).

    The resulting MLP is a transformation :math:`y = f(x)` whose Jacobian entries
    :math:`\frac{\partial y_j}{\partial x_i}` are positive.

    Arguments:
        args: Positional arguments passed to :class:`MLP`.
        kwargs: Keyword arguments passed to :class:`MLP`.

    Example:
        >>> net = MonotonicMLP(3, 4, [16, 32])
        >>> net
        MonotonicMLP(
          (0): MonotonicLinear(in_features=3, out_features=16, bias=True)
          (1): TwoWayELU(alpha=1.0)
          (2): MonotonicLinear(in_features=16, out_features=32, bias=True)
          (3): TwoWayELU(alpha=1.0)
          (4): MonotonicLinear(in_features=32, out_features=4, bias=True)
        )
        >>> x = torch.randn(3)
        >>> torch.autograd.functional.jacobian(net, x)
        tensor([[1.0492, 1.3094, 1.1711],
                [1.1201, 1.3825, 1.2711],
                [0.9397, 1.1915, 1.0787],
                [1.1049, 1.3635, 1.2592]])
    """

    def __init__(self, *args, **kwargs):
        kwargs["activation"] = TwoWayELU
        kwargs["normalize"] = False

        super().__init__(*args, **kwargs)

        for layer in self:
            if isinstance(layer, Linear):
                layer.__class__ = MonotonicLinear
