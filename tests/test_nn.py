r"""Tests for the zuko.nn module."""

import math
import pytest
import torch
import torch.nn as nn

from torch import randn
from typing import Sequence

from zuko.nn import *


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch", [(), (256,)])
def test_Linear(bias: bool, batch: Sequence[int]):
    net = Linear(3, 5, bias=bias)

    x = randn(*batch, 3)
    y = net(x)

    assert y.shape == (*batch, 5)
    assert y.requires_grad


@pytest.mark.parametrize("activation", [None, torch.nn.ELU])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("batch", [(), (256,)])
def test_MLP(activation: type, normalize: bool, batch: Sequence[int]):
    net = MLP(3, 5, activation=activation, normalize=normalize)

    x = randn(*batch, 3)
    y = net(x)

    assert y.shape == (*batch, 5)
    assert y.requires_grad


@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("batch", [(), (256,)])
def test_MaskedMLP(residual: bool, batch: Sequence[int]):
    adjacency = randn(5, 3) < 0
    net = MaskedMLP(adjacency, activation=nn.ELU, residual=residual)

    x = randn(*batch, 3)
    y = net(x)

    assert y.shape == (*batch, 5)
    assert y.requires_grad

    # Jacobian
    x = randn(*batch, 3)
    J = torch.autograd.functional.jacobian(net, x)
    J = J.movedim(len(batch), -2)

    mask = torch.eye(math.prod(batch), dtype=bool)
    mask = mask.reshape(batch + batch)

    assert (J[mask][..., ~adjacency] == 0).all()
    assert (J[~mask] == 0).all()


@pytest.mark.parametrize("batch", [(), (256,)])
def test_MonotonicMLP(batch: Sequence[int]):
    net = MonotonicMLP(3, 5)

    x = randn(*batch, 3)
    y = net(x)

    assert y.shape == (*batch, 5)
    assert y.requires_grad

    # Jacobian
    x = randn(*batch, 3)
    J = torch.autograd.functional.jacobian(net, x)
    J = J.movedim(len(batch), -2)

    mask = torch.eye(math.prod(batch), dtype=bool)
    mask = mask.reshape(batch + batch)

    assert (J[mask] > 0).all()
    assert (J[~mask] == 0).all()
