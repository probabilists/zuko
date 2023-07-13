r"""Tests for the zuko.utils module."""

import math
import pytest
import torch

from torch import randn
from zuko.utils import *


def test_bisection():
    f = torch.cos
    y = torch.tensor(0.0)
    a = torch.rand(256, 1) + 2.0
    b = torch.rand(16)

    x = bisection(f, y, a, b, n=18)

    assert x.shape == (256, 16)
    assert torch.allclose(x, torch.tensor(math.pi / 2), atol=1e-4)
    assert torch.allclose(f(x), y, atol=1e-4)


def test_broadcast():
    # Trivial
    a = randn(2, 3)
    (b,) = broadcast(a)

    assert a.shape == b.shape
    assert (a == b).all()

    # Standard
    a, b, c = randn(1).squeeze(), randn(2), randn(3, 1)
    d, e, f = broadcast(a, b, c)

    assert d.shape == e.shape == f.shape == (3, 2)
    assert (a == d).all() and (b == e).all() and (c == f).all()

    # Invalid
    with pytest.raises(RuntimeError):
        a, b = randn(2), randn(3)
        d, e = broadcast(a, b)

    # Ignore last dimension
    a, b = randn(2), randn(3, 4)
    c, d = broadcast(a, b, ignore=1)

    assert c.shape == (3, 2) and d.shape == (3, 4)
    assert (a == c).all() and (b == d).all()

    # Ignore mixed dimensions
    a, b = randn(2, 3), randn(3, 4)
    c, d = broadcast(a, b, ignore=[0, 1])

    assert c.shape == (2, 3) and d.shape == (2, 3, 4)
    assert (a == c).all() and (b == d).all()


def test_gauss_legendre():
    # Polynomial
    f = lambda x: x**5 - x**2
    F = lambda x: x**6 / 6 - x**3 / 3
    a, b = randn(2, 256, requires_grad=True)

    area = gauss_legendre(f, a, b, n=3)

    assert torch.allclose(area, F(b) - F(a), atol=1e-4)

    # Gradients
    grad_a, grad_b = torch.autograd.grad(area.sum(), (a, b))

    assert torch.allclose(grad_a, -f(a), atol=1e-4)
    assert torch.allclose(grad_b, f(b), atol=1e-4)


def test_odeint():
    # Linear
    alpha = torch.tensor(1.0, requires_grad=True)
    t = torch.tensor(3.0, requires_grad=True)

    f = lambda t, x: -alpha * x
    F = lambda t, x: x * (-alpha * t).exp()

    x0 = randn(256, 1, requires_grad=True)
    xt = odeint(f, x0, torch.zeros_like(t), t, phi=(alpha,))

    assert xt.shape == x0.shape
    assert torch.allclose(xt, F(t, x0), atol=1e-4)

    # Gradients
    grad_x0, grad_t, grad_alpha = torch.autograd.grad(xt.sum(), (x0, t, alpha))
    g_x0, g_t, g_alpha = torch.autograd.grad(F(t, x0).sum(), (x0, t, alpha))

    assert torch.allclose(grad_x0, g_x0, atol=1e-4)
    assert torch.allclose(grad_t, g_t, atol=1e-4)
    assert torch.allclose(grad_alpha, g_alpha, atol=1e-4)


def test_unpack():
    # Normal
    x = randn(26)
    y, z = unpack(x, ((1, 2, 3), (4, 5)))

    assert y.shape == (1, 2, 3)
    assert z.shape == (4, 5)

    # Batched
    x = randn(7, 26)
    y, z = unpack(x, ((1, 2, 3), (4, 5)))

    assert y.shape == (7, 1, 2, 3)
    assert z.shape == (7, 4, 5)
