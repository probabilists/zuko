r"""Tests for the zuko.flows module."""

import pytest
import torch

from functools import partial
from pathlib import Path
from torch import randn

from zuko.flows import *


@pytest.mark.parametrize("F", [NICE, MAF, NSF, SOSPF, NAF, UNAF, CNF, GF, BPF])
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
    x, c = randn(256, 3), randn(256, 5)
    t = flow(c).transform
    z = t.inv(t(x))

    assert torch.allclose(x, z, atol=1e-4)

    # Saving
    torch.save(flow, tmp_path / "flow.pth")

    # Loading
    flow_bis = torch.load(tmp_path / "flow.pth", weights_only=False)

    x, c = randn(3), randn(5)

    seed = torch.seed()
    log_p = flow(c).log_prob(x)
    torch.manual_seed(seed)
    log_p_bis = flow_bis(c).log_prob(x)

    assert torch.allclose(log_p, log_p_bis)

    # Printing
    assert repr(flow)


def test_triangular_transforms():
    order = torch.randperm(5)

    adjacency = torch.rand((5, 5)) < 0.25
    adjacency = adjacency + torch.eye(5, dtype=bool)
    adjacency = torch.tril(adjacency)
    adjacency[1, 0] = True
    adjacency = adjacency[order, :][:, order]

    Ts = [
        ElementWiseTransform,
        GeneralCouplingTransform,
        partial(GeneralCouplingTransform, mask=order % 2),
        MaskedAutoregressiveTransform,
        partial(MaskedAutoregressiveTransform, passes=2),
        partial(MaskedAutoregressiveTransform, order=order),
        partial(MaskedAutoregressiveTransform, adjacency=adjacency),
    ]

    for T in Ts:
        # Without context
        t = T(5)
        x = randn(64, 5)
        y = t()(x)

        assert y.shape == x.shape, T
        assert y.requires_grad, T
        assert torch.allclose(t().inv(y), x, atol=1e-4), T

        # With context
        t = T(5, 7)
        x, c = randn(64, 5), randn(7)
        y = t(c)(x)

        assert y.shape == x.shape, T
        assert y.requires_grad, T
        assert torch.allclose(t(c).inv(y), x, atol=1e-4), T

        # Jacobian
        t = T(5)
        x = randn(5)
        y = t()(x)

        J = torch.autograd.functional.jacobian(t(), x)
        ladj = torch.linalg.slogdet(J).logabsdet

        assert torch.allclose(t().log_abs_det_jacobian(x, y), ladj, atol=1e-4), T
        assert torch.allclose(J.diag().abs().log().sum(), ladj, atol=1e-4), T


def test_adjacency_matrix():
    T = MaskedAutoregressiveTransform

    # With a valid adjacency matrix
    order = torch.randperm(5)

    adjacency = torch.rand((5, 5)) < 0.25
    adjacency = adjacency + torch.eye(5, dtype=bool)
    adjacency = torch.tril(adjacency)
    adjacency[1, 0] = True
    adjacency = adjacency[order, :][:, order]

    t = T(5, adjacency=adjacency)
    x = randn(5)

    J = torch.autograd.functional.jacobian(t(), x)

    assert (J[~adjacency] == 0).all()

    # With False in the diagonal
    adjacency_invalid = adjacency.clone()
    adjacency_invalid[0, 0] = False

    with pytest.raises(AssertionError, match="'adjacency' should have ones on the diagonal."):
        t = T(5, adjacency=adjacency_invalid)

    # With cycles
    adjacency_invalid = adjacency.clone()
    adjacency_invalid[0, 1] = True
    adjacency_invalid[1, 0] = True

    with pytest.raises(AssertionError, match="The graph contains cycles."):
        t = T(5, adjacency=adjacency_invalid)


def test_context_adjacency_matrix():
    T = MaskedAutoregressiveTransform

    # With a valid adjacency matrix
    order = torch.randperm(5)
    adjacency = torch.rand((5, 5)) < 0.25
    adjacency = adjacency + torch.eye(5, dtype=bool)
    adjacency = torch.tril(adjacency)
    adjacency = adjacency[order, :][:, order]

    # context adjacency
    adjacency_context = torch.rand((5, 2)) < 0.25
    adjacency_valid = torch.cat((adjacency, adjacency_context), dim=1)

    t = T(features=5, context=2, adjacency=adjacency_valid)

    x, c = randn(5), randn(2)
    y = t(c)(x)

    assert y.shape == x.shape
    assert y.requires_grad
    assert torch.allclose(t(c).inv(y), x, atol=1e-4)

    J = torch.autograd.functional.jacobian(t(c), x)

    assert (J[~adjacency] == 0).all()

    ladj = torch.linalg.slogdet(J).logabsdet

    assert torch.allclose(t(c).log_abs_det_jacobian(x, y), ladj, atol=1e-4)
    assert torch.allclose(J.diag().abs().log().sum(), ladj, atol=1e-4)

    adjacency_invalid_context = torch.rand((5, 1)) < 0.25
    adjacency_invalid = torch.cat((adjacency, adjacency_invalid_context), dim=1)

    with pytest.raises(AssertionError, match="'adjacency' should have 5 or 7 columns."):
        t = T(features=5, context=2, adjacency=adjacency_invalid)
