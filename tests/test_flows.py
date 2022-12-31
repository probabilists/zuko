r"""Tests for the zuko.flows module."""

import pytest
import torch

from torch import randn
from zuko.flows import *


def test_flows(tmp_path):
    flows = [
        MAF(3, 5),
        NSF(3, 5),
        SOSPF(3, 5),
        NAF(3, 5),
        NAF(3, 5, unconstrained=True),
        CNF(3, 5),
    ]

    for flow in flows:
        # Evaluation of log_prob
        x, y = randn(256, 3), randn(5)
        log_p = flow(y).log_prob(x)

        assert log_p.shape == (256,), flow
        assert log_p.requires_grad, flow

        flow.zero_grad(set_to_none=True)
        loss = -log_p.mean()
        loss.backward()

        for p in flow.parameters():
            assert p.grad is not None, flow

        # Sampling
        x = flow(y).sample((32,))

        assert x.shape == (32, 3), flow

        # Reparameterization trick
        x = flow(y).rsample()

        flow.zero_grad(set_to_none=True)
        loss = x.square().sum().sqrt()
        loss.backward()

        for p in flow.parameters():
            assert p.grad is not None, flow

        # Invertibility
        x, y = randn(256, 3), randn(256, 5)

        transforms = [t(y) for t in flow.transforms]

        z = x

        for t in transforms:
            z = t(z)

        for t in reversed(transforms):
            z = t.inv(z)

        assert torch.allclose(x, z, atol=1e-4), flow

        # Saving
        torch.save(flow, tmp_path / 'flow.pth')

        # Loading
        flow_bis = torch.load(tmp_path / 'flow.pth')

        x, y = randn(3), randn(5)

        seed = torch.seed()
        log_p = flow(y).log_prob(x)
        torch.manual_seed(seed)
        log_p_bis = flow_bis(y).log_prob(x)

        assert torch.allclose(log_p, log_p_bis), flow


def test_autoregressive_transforms():
    ATs = [
        MaskedAutoregressiveTransform,
        NeuralAutoregressiveTransform,
        UnconstrainedNeuralAutoregressiveTransform,
    ]

    for AT in ATs:
        # Without context
        t = AT(3)
        x = randn(3)
        z = t()(x)

        assert z.shape == x.shape, t
        assert z.requires_grad, t
        assert torch.allclose(t().inv(z), x, atol=1e-4), t

        # With context
        t = AT(3, 5)
        x, y = randn(256, 3), randn(5)
        z = t(y)(x)

        assert z.shape == x.shape, t
        assert z.requires_grad, t
        assert torch.allclose(t(y).inv(z), x, atol=1e-4), t

        # Passes

        ## Fully autoregressive
        t = AT(7)
        x = randn(7)
        J = torch.autograd.functional.jacobian(t(), x)

        assert (torch.triu(J, diagonal=1) == 0).all(), t

        ## Coupling
        t = AT(7, passes=2)
        x = randn(7)
        J = torch.autograd.functional.jacobian(t(), x)

        assert (torch.triu(J, diagonal=1) == 0).all(), t
        assert (torch.tril(J[:4, :4], diagonal=-1) == 0).all(), t
        assert (torch.tril(J[4:, 4:], diagonal=-1) == 0).all(), t


def test_Glow(tmp_path):
    flow = Glow((3, 32, 32), context=[5, 0, 5])

    # Evaluation of log_prob
    x, y = randn(8, 3, 32, 32), [randn(5, 16, 16), None, randn(8, 5, 4, 4)]
    log_p = flow(y).log_prob(x)

    assert log_p.shape == (8,)
    assert log_p.requires_grad

    flow.zero_grad(set_to_none=True)
    loss = -log_p.mean()
    loss.backward()

    for p in flow.parameters():
        assert p.grad is not None

    # Sampling
    x = flow(y).sample()

    assert x.shape == (8, 3, 32, 32)

    # Reparameterization trick
    x = flow(y).rsample()

    flow.zero_grad(set_to_none=True)
    loss = x.square().sum().sqrt()
    loss.backward()

    for p in flow.parameters():
        assert p.grad is not None

    # Saving
    torch.save(flow, tmp_path / 'flow.pth')

    # Loading
    flow_bis = torch.load(tmp_path / 'flow.pth')

    x, y = randn(3, 32, 32), [randn(5, 16, 16), None, randn(5, 4, 4)]

    seed = torch.seed()
    log_p = flow(y).log_prob(x)
    torch.manual_seed(seed)
    log_p_bis = flow_bis(y).log_prob(x)

    assert torch.allclose(log_p, log_p_bis)
