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


def test_multivariate_transforms():
    ts = [
        LULinearTransform(randn(3, 3), dim=-2),
        PermutationTransform(torch.randperm(3), dim=-2),
        PixelShuffleTransform(dim=-2),
    ]

    for t in ts:
        # Shapes
        x = randn(256, 3, 8)
        y = t(x)

        assert t.forward_shape(x.shape) == y.shape, t
        assert t.inverse_shape(y.shape) == x.shape, t

        # Inverse
        z = t.inv(y)

        assert x.shape == z.shape, t
        assert torch.allclose(x, z, atol=1e-4), t

        # Jacobian
        x = randn(3, 8)
        y = t(x)

        jacobian = torch.autograd.functional.jacobian(t, x)
        jacobian = jacobian.reshape(3 * 8, 3 * 8)

        _, ladj = torch.slogdet(jacobian)

        assert torch.allclose(t.log_abs_det_jacobian(x, y), ladj, atol=1e-4), t

        # Inverse Jacobian
        z = t.inv(y)

        jacobian = torch.autograd.functional.jacobian(t.inv, y)
        jacobian = jacobian.reshape(3 * 8, 3 * 8)

        _, ladj = torch.slogdet(jacobian)

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


def test_DropTransform():
    dist = Normal(randn(3), abs(randn(3)) + 1)
    t = DropTransform(dist)

    # Call
    x = randn(256, 5)
    y = t(x)

    assert t.forward_shape(x.shape) == y.shape
    assert t.inverse_shape(y.shape) == x.shape

    # Inverse
    z = t.inv(y)

    assert x.shape == z.shape
    assert not torch.allclose(x, z)

    # Jacobian
    ladj = t.log_abs_det_jacobian(x, y)

    assert ladj.shape == (256,)
