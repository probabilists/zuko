r"""Tests for the zuko.nn module."""

import pytest
import torch
import torch.nn as nn

from torch import randn

from zuko.nn import *


@pytest.mark.parametrize("bias", [True, False])
def test_Linear(bias: bool):
    net = Linear(3, 5, bias=True)

    # Non-batched
    x = randn(3)
    y = net(x)

    assert y.shape == (5,)
    assert y.requires_grad

    # Batched
    x = randn(256, 3)
    y = net(x)

    assert y.shape == (256, 5)


@pytest.mark.parametrize("activation", [None, torch.nn.ELU])
@pytest.mark.parametrize("normalize", [True, False])
def test_MLP(activation: callable, normalize: bool):
    net = MLP(3, 5, activation=activation, normalize=normalize)

    # Non-batched
    x = randn(3)
    y = net(x)

    assert y.shape == (5,)
    assert y.requires_grad

    # Batched
    x = randn(256, 3)
    y = net(x)

    assert y.shape == (256, 5)


@pytest.mark.parametrize("residual", [True, False])
def test_MaskedMLP(residual: bool):
    adjacency = randn(5, 3) < 0
    net = MaskedMLP(adjacency, activation=nn.ELU, residual=residual)

    # Non-batched
    x = randn(3)
    y = net(x)

    assert y.shape == (5,)
    assert y.requires_grad

    # Batched
    x = randn(256, 3)
    y = net(x)

    assert y.shape == (256, 5)

    # Jacobian
    x = randn(3)
    J = torch.autograd.functional.jacobian(net, x)

    assert (J[~adjacency] == 0).all()


def test_MonotonicMLP():
    net = MonotonicMLP(3, 5)

    # Non-batched
    x = randn(3)
    y = net(x)

    assert y.shape == (5,)
    assert y.requires_grad

    # Batched
    x = randn(256, 3)
    y = net(x)

    assert y.shape == (256, 5)

    # Jacobian
    x = randn(3)
    J = torch.autograd.functional.jacobian(net, x)

    assert (J >= 0).all()
