r"""Neural networks, layers and modules."""

__all__ = ['MLP', 'MaskedMLP', 'MonotonicMLP', 'FCN']

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor, BoolTensor
from typing import *


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

        self.dim = dim if type(dim) is int else tuple(dim)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        variance, mean = torch.var_mean(x, unbiased=True, dim=self.dim, keepdim=True)

        return (x - mean) / (variance + self.eps).sqrt()


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
        normalize: bool = False,
        **kwargs,
    ):
        if activation is None:
            activation = nn.ReLU

        normalization = LayerNorm if normalize else lambda: None

        layers = []

        for before, after in zip(
            [in_features] + hidden_features,
            hidden_features + [out_features],
        ):
            layers.extend([
                nn.Linear(before, after, **kwargs),
                activation(),
                normalization(),
            ])

        layers = layers[:-2]
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


class MaskedMLP(nn.Sequential):
    r"""Creates a masked multi-layer perceptron (MaskedMLP).

    The resulting MLP is a transformation :math:`y = f(x)` whose Jacobian entries
    :math:`\frac{\partial y_i}{\partial x_j}` are null if :math:`A_{ij} = 0`.

    Arguments:
        adjacency: The adjacency matrix :math:`A \in \{0, 1\}^{M \times N}`.
        hidden_features: The numbers of hidden features.
        activation: The activation function constructor. If :py:`None`, use
            :class:`torch.nn.ReLU` instead.

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

    def __init__(
        self,
        adjacency: BoolTensor,
        hidden_features: List[int] = [64, 64],
        activation: Callable[[], nn.Module] = None,
    ):
        out_features, in_features = adjacency.shape

        if activation is None:
            activation = nn.ReLU

        # Merge outputs with the same dependencies
        adjacency, inverse = torch.unique(adjacency, dim=0, return_inverse=True)

        # P_ij = 1 if A_ik = 1 for all k such that A_jk = 1
        precedence = adjacency.int() @ adjacency.int().t() == adjacency.sum(dim=-1)

        # Layers
        layers = []

        for i, features in enumerate(hidden_features + [out_features]):
            if i > 0:
                mask = precedence[:, indices]
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

            layers.extend([
                MaskedLinear(adjacency=mask),
                activation(),
            ])

        layers = layers[:-1]

        super().__init__(*layers)

        self.in_features = in_features
        self.out_features = out_features


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
        kwargs['normalize'] = False

        super().__init__(*args, **kwargs)

        for i, layer in enumerate(self):
            if isinstance(layer, nn.Linear):
                layer.__class__ = MonotonicLinear
            elif isinstance(layer, nn.ELU):
                layer.__class__ = TwoWayELU


class FCN(nn.Sequential):
    r"""Creates a fully convolutional neural network (FCN).

    The architecture is inspired by ConvNeXt blocks which mix depthwise and 1 by 1
    convolutions to improve the efficiency/accuracy trade-off.

    References:
        | A ConvNet for the 2020s (Lui et al., 2022)
        | https://arxiv.org/abs/2201.03545

    Arguments:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        hidden_channels: The number of hidden channels.
        hidden_blocks: The number of hidden blocks. Each block consists in an optional
            normalization, a depthwise convolution, an activation and a 1 by 1
            convolution.
        kernel_size: The size of the convolution kernels.
        activation: The activation function constructor. If :py:`None`, use
            :class:`torch.nn.ReLU` instead.
        normalize: Whether channels are normalized or not.
        spatial: The number of spatial dimensions. Can be either 1, 2 or 3.
        kwargs: Keyword arguments passed to :class:`torch.nn.Conv2d`.

    Example:
        >>> net = FCN(3, 16, 64, activation=nn.ELU)
        >>> net
        FCN(
          (0): Conv2d(3, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
          (1): Conv2d(64, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=64)
          (2): ELU(alpha=1.0)
          (3): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
          (4): Conv2d(64, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=64)
          (5): ELU(alpha=1.0)
          (6): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
          (7): Conv2d(64, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=64)
          (8): ELU(alpha=1.0)
          (9): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
          (10): Conv2d(64, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        )
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        hidden_blocks: int = 3,
        kernel_size: int = 5,
        activation: Callable[[], nn.Module] = None,
        normalize: bool = False,
        spatial: int = 2,
        **kwargs,
    ):
        # Components
        convolution = {
            1: nn.Conv1d,
            2: nn.Conv2d,
            3: nn.Conv3d,
        }.get(spatial)

        if activation is None:
            activation = nn.ReLU

        if normalize:
            normalization = lambda: LayerNorm(dim=-(spatial + 1))
        else:
            normalization = lambda: None

        layers = [
            convolution(
                in_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                **kwargs,
            )
        ]

        for i in range(hidden_blocks):
            layers.extend([
                normalization(),
                convolution(
                    hidden_channels,
                    hidden_channels * 4,
                    groups=hidden_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    **kwargs,
                ),
                activation(),
                convolution(hidden_channels * 4, hidden_channels, kernel_size=1),
            ])

        layers.append(
            convolution(
                hidden_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                **kwargs,
            )
        )

        layers = filter(lambda l: l is not None, layers)

        super().__init__(*layers)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial = spatial

    def forward(self, x: Tensor) -> Tensor:
        dim = -(self.spatial + 1)
        batch_shape = x.shape[:dim]

        x = x.reshape(-1, *x.shape[dim:])
        y = super().forward(x)
        y = y.reshape(*batch_shape, *y.shape[dim:])

        return y
