r"""Tests for the zuko.nn module."""

import pytest
import torch
import torch.nn as nn

from torch import randn

from zuko.nn import *


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_Linear(bias: bool, device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("No CUDA devices available")

    torch.set_default_device(device)

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
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_MLP(activation: callable, normalize: bool, device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("No CUDA devices available")

    torch.set_default_device(device)

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
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_MaskedMLP(residual: bool, device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("No CUDA devices available")

    torch.set_default_device(device)

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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_MonotonicMLP(device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("No CUDA devices available")

    torch.set_default_device(device)

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
