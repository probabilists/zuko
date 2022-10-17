r"""Tests for the zuko.utils module."""

import math
import pytest
import torch

from torch import rand
from zuko.utils import *


def test_bisection():
    f = torch.cos
    a = 2.0 + rand(16, 1)
    b = rand(8)

    x = bisection(f, a, b, n=21)

    assert x.shape == (16, 8)
    assert torch.allclose(x, torch.tensor(math.pi / 2), atol=1e-5)
    assert torch.allclose(f(x), torch.tensor(0.0), atol=1e-5)


def test_broadcast():
    # Trivial
    a = rand(2, 3)
    (b,) = broadcast(a)

    assert a.shape == b.shape
    assert (a == b).all()

    # Standard
    a, b, c = rand(1).squeeze(), rand(2), rand(3, 1)
    d, e, f = broadcast(a, b, c)

    assert d.shape == e.shape == f.shape == (3, 2)
    assert (a == d).all() and (b == e).all() and (c == f).all()

    # Invalid
    with pytest.raises(RuntimeError):
        a, b = rand(2), rand(3)
        d, e = broadcast(a, b)

    # Ignore last dimension
    a, b = rand(2), rand(3, 4)
    c, d = broadcast(a, b, ignore=1)

    assert c.shape == (3, 2) and d.shape == (3, 4)
    assert (a == c).all() and (b == d).all()

    # Ignore mixed dimensions
    a, b = rand(2, 3), rand(3, 4)
    c, d = broadcast(a, b, ignore=[0, 1])

    assert c.shape == (2, 3) and d.shape == (2, 3, 4)
    assert (a == c).all() and (b == d).all()


def test_gauss_legendre():
    # Polynomial
    f = lambda x: x**5 - x**2
    F = lambda x: x**6 / 6 - x**3 / 3
    a, b = 5 * rand(2, 64)

    area = gauss_legendre(f, a, b, n=3)

    assert torch.allclose(F(b) - F(a), area, atol=1e-5, rtol=1e-3)

    # Gradients
    grad_a, grad_b = torch.autograd.functional.jacobian(
        lambda a, b: gauss_legendre(f, a, b).sum(),
        (a, b),
    )

    assert torch.allclose(-f(a), grad_a)
    assert torch.allclose(f(b), grad_b)
