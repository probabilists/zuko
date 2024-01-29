r"""Tests for the zuko.flows module."""

import pytest
import torch

from functools import partial
from pathlib import Path
from torch import randn
from zuko.flows import *

torch.set_default_dtype(torch.float64)


@pytest.mark.parametrize('F', [GMM, NICE, MAF, NSF, SOSPF, NAF, UNAF, CNF, GF, BPF])
def test_flows(tmp_path: Path, F: callable):
    flow = F(3, 5)

    # Evaluation of log_prob
    x, c = randn(256, 3), randn(5)
    log_p = flow(c).log_prob(x)

    assert log_p.shape == (256,)
    assert log_p.requires_grad

    flow.zero_grad(set_to_none=True)
    loss = -log_p.mean()
    loss.backward()

    for p in flow.parameters():
        assert p.grad is not None

    # Sampling
    x = flow(c).sample((32,))

    assert x.shape == (32, 3)

    # Reparameterization trick
    if flow(c).has_rsample:
        x = flow(c).rsample()

        flow.zero_grad(set_to_none=True)
        loss = x.square().sum().sqrt()
        loss.backward()

        for p in flow.parameters():
            assert p.grad is not None

    # Invertibility
    if isinstance(flow, Flow):
        x, c = randn(256, 3), randn(256, 5)
        t = flow(c).transform
        z = t.inv(t(x))

        assert torch.allclose(x, z, atol=1e-4)

    # Saving
    torch.save(flow, tmp_path / 'flow.pth')

    # Loading
    flow_bis = torch.load(tmp_path / 'flow.pth')

    x, c = randn(3), randn(5)

    seed = torch.seed()
    log_p = flow(c).log_prob(x)
    torch.manual_seed(seed)
    log_p_bis = flow_bis(c).log_prob(x)

    assert torch.allclose(log_p, log_p_bis)

    # Printing
    assert repr(flow)


def test_triangular_transforms():
    Ts = [
        ElementWiseTransform,
        GeneralCouplingTransform,
        MaskedAutoregressiveTransform,
        partial(MaskedAutoregressiveTransform, passes=2),
    ]

    for T in Ts:
        # Without context
        t = T(3)
        x = randn(3)
        y = t()(x)

        assert y.shape == x.shape, t
        assert y.requires_grad, t
        assert torch.allclose(t().inv(y), x, atol=1e-4), T

        # With context
        t = T(3, 5)
        x, c = randn(256, 3), randn(5)
        y = t(c)(x)

        assert y.shape == x.shape, T
        assert y.requires_grad, T
        assert torch.allclose(t(c).inv(y), x, atol=1e-4), T

        # Jacobian
        t = T(7)
        x = randn(7)
        y = t()(x)

        J = torch.autograd.functional.jacobian(t(), x)
        ladj = torch.linalg.slogdet(J).logabsdet

        assert torch.allclose(t().log_abs_det_jacobian(x, y), ladj, atol=1e-4), T
        assert torch.allclose(J.diag().abs().log().sum(), ladj, atol=1e-4), T
