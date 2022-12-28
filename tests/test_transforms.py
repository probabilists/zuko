r"""Tests for the zuko.transforms module."""

import pytest
import torch

from torch import randn
from torch.distributions import *
from zuko.transforms import *


def test_univariate_transforms():
    ts = [
        IdentityTransform(),
        CosTransform(),
        SinTransform(),
        SoftclipTransform(),
        MonotonicAffineTransform(randn(256), randn(256)),
        MonotonicRQSTransform(randn(256, 8), randn(256, 8), randn(256, 7)),
        MonotonicTransform(lambda x: x**3),
        UnconstrainedMonotonicTransform(lambda x: torch.exp(-x**2) + 1e-2, randn(256)),
        SOSPolynomialTransform(randn(256, 2, 4), randn(256)),
    ]

    for t in ts:
        # Call
        if hasattr(t.domain, 'lower_bound'):
            x = torch.linspace(t.domain.lower_bound + 1e-2, t.domain.upper_bound - 1e-2, 256)
        else:
            x = torch.linspace(-5.0, 5.0, 256)

        y = t(x)

        assert x.shape == y.shape, t

        # Inverse
        z = t.inv(y)

        assert torch.allclose(x, z, atol=1e-4), t

        # Jacobian
        J = torch.autograd.functional.jacobian(t, x)

        assert (torch.triu(J, diagonal=1) == 0).all(), t
        assert (torch.tril(J, diagonal=-1) == 0).all(), t

        ladj = torch.diag(J).abs().log()

        assert torch.allclose(t.log_abs_det_jacobian(x, y), ladj, atol=1e-4), t

        # Inverse Jacobian
        J = torch.autograd.functional.jacobian(t.inv, y)

        assert (torch.triu(J, diagonal=1) == 0).all(), t
        assert (torch.tril(J, diagonal=-1) == 0).all(), t

        ladj = torch.diag(J).abs().log()

        assert torch.allclose(t.inv.log_abs_det_jacobian(y, z), ladj, atol=1e-4), t


def test_FFJTransform():
    a = torch.randn(3)
    f = lambda x, t: a * x
    t = FFJTransform(f, time=torch.tensor(1.0))

    # Call
    x = randn(256, 3)
    y = t(x)

    assert x.shape == y.shape

    # Inverse
    z = t.inv(y)

    assert torch.allclose(x, z, atol=1e-4)

    # Jacobian
    ladj = t.log_abs_det_jacobian(x, y)

    assert ladj.shape == x.shape[:-1]


def test_PermutationTransform():
    t = PermutationTransform(torch.randperm(8))

    x = torch.randn(256, 8)
    y = t(x)

    assert x.shape == y.shape

    match = x[:, :, None] == y[:, None, :]

    assert (match.sum(dim=-1) == 1).all()
    assert (match.sum(dim=-2) == 1).all()

    z = t.inv(y)

    assert x.shape == z.shape
    assert (x == z).all()
