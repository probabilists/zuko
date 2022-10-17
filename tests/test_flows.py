r"""Tests for the zuko.flows module."""

import pytest
import torch

from torch import randn
from zuko.flows import *
from zuko.transforms import SoftclipTransform


def test_flows(tmp_path):
    flows = [
        MAF(3, 5),
        NSF(3, 5),
        NAF(3, 5),
        NAF(3, 5, unconstrained=True),
    ]

    for flow in flows:
        # Evaluation of log_prob
        x, y = randn(256, 3), randn(5)
        log_p = flow(y).log_prob(x)

        assert log_p.shape == (256,), flow
        assert log_p.requires_grad, flow

        loss = -log_p.mean()
        loss.backward()

        for p in flow.parameters():
            assert hasattr(p, 'grad'), flow

        # Sampling
        z = flow(y).sample((32,))

        assert z.shape == (32, 3), flow

        # Invertibility
        x, y = randn(256, 3), randn(5)

        transforms = [t(y) for t in flow.transforms]

        z = x

        for t in transforms:
            z = t(z)

        for t in reversed(transforms):
            z = t.inv(z)

        assert torch.allclose(x, z, atol=1e-5), flow

        # Saving
        torch.save(flow, tmp_path / 'flow.pth')

        # Loading
        flow_bis = torch.load(tmp_path / 'flow.pth')

        x, y = randn(3), randn(5)

        log_p = flow(y).log_prob(x)
        log_p_bis = flow_bis(y).log_prob(x)

        assert torch.allclose(log_p, log_p_bis), flow


def test_autoregressive_transforms():
    softclip = SoftclipTransform()

    ATs = [
        MaskedAutoregressiveTransform,
        NeuralAutoregressiveTransform,
        UnconstrainedNeuralAutoregressiveTransform,
    ]

    for AT in ATs:
        # Without context
        t = AT(3)
        x = softclip(randn(3))
        z = t()(x)

        assert z.shape == x.shape, t
        assert z.requires_grad, t
        assert torch.allclose(t().inv(z), x, atol=1e-5), t

        # With context
        t = AT(3, 5)
        x, y = softclip(randn(256, 3)), randn(5)
        z = t(y)(x)

        assert z.shape == x.shape, t
        assert z.requires_grad, t
        assert torch.allclose(t(y).inv(z), x, atol=1e-5), t

        # Passes

        ## Fully autoregressive
        t = AT(7)
        x = softclip(randn(7))
        J = torch.autograd.functional.jacobian(t(), x)

        assert (torch.triu(J, diagonal=1) == 0).all(), t

        ## Coupling
        t = AT(7, passes=2)
        x = softclip(randn(7))
        J = torch.autograd.functional.jacobian(t(), x)

        assert (torch.triu(J, diagonal=1) == 0).all(), t
        assert (torch.tril(J[:4, :4], diagonal=-1) == 0).all(), t
        assert (torch.tril(J[4:, 4:], diagonal=-1) == 0).all(), t
