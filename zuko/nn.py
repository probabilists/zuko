r"""Neural networks, layers and modules."""

__all__ = ['MLP', 'MaskedMLP', 'MonotonicMLP']

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor, BoolTensor
from typing import *


class BatchNorm0d(nn.BatchNorm1d):
    r"""Creates a batch normalization (BatchNorm) layer for scalars.

    References:
        | Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (Ioffe et al., 2015)
        | https://arxiv.org/abs/1502.03167

    Arguments:
        args: Positional arguments passed to :class:`torch.nn.BatchNorm1d`.
        kwargs: Keyword arguments passed to :class:`torch.nn.BatchNorm1d`.
    """

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape

        x = x.reshape(-1, shape[-1])
        x = super().forward(x)
        x = x.reshape(shape)

        return x


class MLP(nn.Sequential):
    r"""Creates a multi-layer perceptron (MLP).

    Also known as fully connected feedforward network, an MLP is a series of
    non-linear parametric transformations

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
        activation: The activation function. If :py:`None`, use :class:`torch.nn.ReLU`
            instead.
        batchnorm: Whether to use batch normalization or not.
        dropout: The dropout rate.
        kwargs: Keyword arguments passed to :class:`torch.nn.Linear`.

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
        hidden_features: List[int] = [64, 64],
        activation: Callable[[], nn.Module] = None,
        batchnorm: bool = False,
        dropout: float = 0.0,
        **kwargs,
    ):
        if activation is None:
            activation = nn.ReLU

        batchnorm = BatchNorm0d if batchnorm else lambda _: None
        dropout = nn.Dropout(dropout) if dropout > 0 else None

        layers = []

        for before, after in zip(
            [in_features] + hidden_features,
            hidden_features + [out_features],
        ):
            layers.extend([
                nn.Linear(before, after, **kwargs),
                batchnorm(after),
                activation(),
                dropout,
            ])

        layers = layers[:-3]
        layers = filter(lambda l: l is not None, layers)

        super().__init__(*layers)

        self.in_features = in_features
        self.out_features = out_features


class MaskedLinear(nn.Linear):
    r"""Creates a masked linear layer.

    .. math:: y = x (W \odot A)^T + b

    Arguments:
        adjacency: The adjacency matrix :math:`A \in \{0, 1\}^{M \times N}`.
        kwargs: Keyword arguments passed to :class:`torch.nn.Linear`.
    """

    def __init__(self, adjacency: BoolTensor, **kwargs):
        super().__init__(*reversed(adjacency.shape), **kwargs)

        self.register_buffer('mask', adjacency)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.mask * self.weight, self.bias)


class MaskedMLP(MLP):
    r"""Creates a masked multi-layer perceptron (MaskedMLP).

    The resulting MLP is a transformation :math:`y = f(x)` whose Jacobian entries
    :math:`\frac{\partial y_i}{\partial x_j}` are null if :math:`A_{ij} = 0`.

    Arguments:
        adjacency: The adjacency matrix :math:`A \in \{0, 1\}^{M \times N}`.
        args: Positional arguments passed to :class:`MLP`.
        kwargs: Keyword arguments passed to :class:`MLP`.

    Example:
        >>> adjacency = torch.randn(4, 3) < 0
        >>> adjacency
        tensor([[False,  True, False],
                [ True, False,  True],
                [False,  True, False],
                [False,  True,  True]])
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
        tensor([[ 0.0000,  0.0031,  0.0000],
                [-0.0323,  0.0000, -0.0547],
                [ 0.0000, -0.0245,  0.0000],
                [ 0.0000,  0.0060, -0.0063]])
    """

    def __init__(self, adjacency: BoolTensor, *args, **kwargs):
        super().__init__(*reversed(adjacency.shape), *args, **kwargs)

        # j precedes i if A_ik = 1 for all k such that A_jk = 1
        precedence = adjacency.int() @ adjacency.int().t() == adjacency.sum(dim=-1)

        for i, layer in enumerate(self):
            if isinstance(layer, nn.Linear):
                if i > 0:
                    mask = precedence[:, indices]
                else:
                    mask = adjacency

                if (~mask).all():
                    raise ValueError("The adjacency matrix leads to a null Jacobian.")

                if i < len(self) - 1:
                    reachable = mask.sum(dim=-1).nonzero().squeeze()
                    indices = reachable[torch.arange(layer.out_features) % len(reachable)]
                    mask = mask[indices]

                self[i] = MaskedLinear(adjacency=mask)


class MonotonicLinear(nn.Linear):
    r"""Creates a monotonic linear layer.

    .. math:: y = x |W|^T + b

    Arguments:
        args: Positional arguments passed to :class:`torch.nn.Linear`.
        kwargs: Keyword arguments passed to :class:`torch.nn.Linear`.
    """

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.abs(), self.bias)


class TwoWayELU(nn.ELU):
    r"""Creates a layer that splits the input into two groups and applies
    :math:`\text{ELU}(x)` to the first and :math:`-\text{ELU}(-x)` to the second.

    Arguments:
        args: Positional arguments passed to :class:`torch.nn.ELU`.
        kwargs: Keyword arguments passed to :class:`torch.nn.ELU`.
    """

    def forward(self, x: Tensor) -> Tensor:
        x0, x1 = torch.chunk(x, 2, dim=-1)

        return torch.cat((
            super().forward(x0),
            -super().forward(-x1),
        ), dim=-1)


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
        tensor([[0.8742, 0.9439, 0.9759],
                [0.8969, 0.9716, 0.9866],
                [1.0780, 1.1651, 1.2056],
                [0.8596, 0.9400, 0.9502]])
    """

    def __init__(self, *args, **kwargs):
        kwargs['activation'] = nn.ELU
        kwargs['batchnorm'] = False

        super().__init__(*args, **kwargs)

        for i, layer in enumerate(self):
            if isinstance(layer, nn.Linear):
                layer.__class__ = MonotonicLinear
            elif isinstance(layer, nn.ELU):
                layer.__class__ = TwoWayELU
