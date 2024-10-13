r"""Tests for the zuko.transforms module."""

import pytest
import torch

from torch import randn
from torch.distributions import *

from zuko.transforms import *


@pytest.mark.parametrize("batched", [False, True])
def test_univariate_transforms(batched: bool):
    batch = (256,) if batched else ()

    ts = [
        IdentityTransform(),
        CosTransform(),
        SinTransform(),
        SoftclipTransform(),
        CircularShiftTransform(),
        SignedPowerTransform(randn(batch)),
        MonotonicAffineTransform(randn(batch), randn(batch)),
        MonotonicRQSTransform(randn(*batch, 8), randn(*batch, 8), randn(*batch, 7)),
        MonotonicTransform(lambda x: x**3),
        BernsteinTransform(randn(*batch, 16)),
        BoundedBernsteinTransform(randn(*batch, 16)),
        GaussianizationTransform(randn(*batch, 8), randn(*batch, 8)),
        UnconstrainedMonotonicTransform(lambda x: torch.exp(-(x**2)) + 1e-2, randn(batch)),
        SOSPolynomialTransform(randn(*batch, 3, 5), randn(batch)),
    ]

    for t in ts:
        # Call
        if hasattr(t.domain, "lower_bound"):
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

        assert (torch.triu(J, diagonal=+1) == 0).all(), t
        assert (torch.tril(J, diagonal=-1) == 0).all(), t

        ladj = torch.diag(J).abs().log()

        assert torch.allclose(t.log_abs_det_jacobian(x, y), ladj, atol=1e-4), t

        # Compound call
        y_comp, ladj_comp = t.call_and_ladj(x)

        assert torch.allclose(y_comp, y, atol=1e-4), t
        assert torch.allclose(ladj_comp, ladj, atol=1e-4), t

        # Inverse Jacobian
        J = torch.autograd.functional.jacobian(t.inv, y)

        assert (torch.triu(J, diagonal=+1) == 0).all(), t
        assert (torch.tril(J, diagonal=-1) == 0).all(), t

        ladj = torch.diag(J).abs().log()

        assert torch.allclose(t.inv.log_abs_det_jacobian(y, z), ladj, atol=1e-4), t


def test_multivariate_transforms():
    A, B = torch.randn(5, 16), torch.randn(16, 5)
    f = lambda t, x: torch.sigmoid(x @ A) @ B

    ts = [
        FreeFormJacobianTransform(f, 0.0, 1.0, atol=1e-7, rtol=1e-6),
        PermutationTransform(torch.randperm(5)),
        RotationTransform(randn(5, 5)),
        LULinearTransform(randn(5, 5)),
    ]

    for t in ts:
        # Shapes
        x = randn(256, 5)
        y = t(x)

        assert x.shape == y.shape, t

        # Inverse
        z = t.inv(y)

        assert torch.allclose(x, z, atol=1e-4), t

        # Jacobian
        x = randn(5)
        y = t(x)

        J = torch.autograd.functional.jacobian(t, x)
        ladj = torch.linalg.slogdet(J).logabsdet

        assert torch.allclose(t.log_abs_det_jacobian(x, y), ladj, atol=1e-4), t

        # Compound call
        y_comp, ladj_comp = t.call_and_ladj(x)

        assert torch.allclose(y_comp, y, atol=1e-4), t
        assert torch.allclose(ladj_comp, ladj, atol=1e-4), t

        # Inverse Jacobian
        z = t.inv(y)

        J = torch.autograd.functional.jacobian(t.inv, y)
        ladj = torch.linalg.slogdet(J).logabsdet

        assert torch.allclose(t.inv.log_abs_det_jacobian(y, z), ladj, atol=1e-4), t
