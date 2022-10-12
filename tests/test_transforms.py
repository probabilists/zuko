r"""Tests for the zuko.transforms module."""

import pytest
import torch

from zuko.transforms import *


def test_univariate_transforms():
    ts = [
        CosTransform(),
        SinTransform(),
        SoftclipTransform(),
        MonotonicAffineTransform(torch.tensor(42.0), torch.tensor(-0.69)),
        MonotonicRQSTransform(*map(torch.rand, (8, 8, 7))),
        MonotonicTransform(lambda x: x**3),
        UnconstrainedMonotonicTransform(lambda x: torch.exp(-x**2) + 1e-2, torch.tensor(0.0)),
    ]

    for t in ts:
        if hasattr(t.domain, 'lower_bound'):
            x = torch.linspace(t.domain.lower_bound, t.domain.upper_bound, 256)
        else:
            x = torch.linspace(-3.0, 3.0, 256)

        y = t(x)

        assert x.shape == y.shape, t

        z = t.inv(y)

        assert torch.allclose(x, z, atol=1e-5), t

        # Jacobian
        J = torch.autograd.functional.jacobian(t, x)

        assert (torch.triu(J, diagonal=1) == 0).all()
        assert (torch.tril(J, diagonal=-1) == 0).all()

        ladj = torch.diag(J).abs().log()

        assert torch.allclose(ladj, t.log_abs_det_jacobian(x, y), atol=1e-5), t


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
