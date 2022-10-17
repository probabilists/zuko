r"""Tests for the zuko.nn module."""

import pytest
import torch
import torch.nn as nn

from torch import randn
from zuko.nn import *


def test_MLP():
    net = MLP(3, 5)

    # Non-batched
    x = randn(3)
    y = net(x)

    assert y.shape == (5,)
    assert y.requires_grad

    # Batched
    x = randn(256, 3)
    y = net(x)

    assert y.shape == (256, 5)


def test_MaskedMLP():
    adjacency = randn(5, 3) < 0
    net = MaskedMLP(adjacency, activation=nn.ELU)

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
