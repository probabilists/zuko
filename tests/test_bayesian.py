r"""Tests for the zuko.bayesian module."""

import pytest
import torch
import torch.nn as nn

from pathlib import Path
from torch import randn

from zuko.bayesian import BayesianModel
from zuko.flows import *
from zuko.nn import *


@pytest.mark.parametrize("activation", [None, torch.nn.ELU])
@pytest.mark.parametrize("normalize", [True, False])
def test_BayesianMLP(activation: callable, normalize: bool):
    net = MLP(3, 5, activation=activation, normalize=normalize)
    bnet = BayesianModel(net)

    # Non-batched
    x = randn(3)

    # test context manager
    with bnet.sample() as sampled_net:
        y = sampled_net(x)
        assert y.shape == (5,)
        assert y.requires_grad

    # test single sampled model
    sampled_net = bnet.sample_model()
    y = sampled_net(x)
    assert y.shape == (5,)
    assert y.requires_grad

    # Batched
    x = randn(256, 3)
    with bnet.sample() as sampled_net:
        y = sampled_net(x)
        assert y.shape == (256, 5)

    ys = []
    for _ in range(2):
        with bnet.sample() as sampled_net:
            ys.append(sampled_net(x))
    assert not torch.allclose(ys[0], ys[1])

    # compute KL divergence
    kl = bnet.kl_divergence()
    assert kl.item() >= 0


@pytest.mark.parametrize("residual", [True, False])
def test_MaskedBayesianMLP(residual: bool):
    adjacency = randn(5, 3) < 0
    net = MaskedMLP(adjacency, activation=nn.ELU, residual=residual)
    bnet = BayesianModel(net)

    # Non-batched
    x = randn(3)
    y = net(x)

    assert y.shape == (5,)
    assert y.requires_grad

    # test context manager
    with bnet.sample() as sampled_net:
        y = sampled_net(x)
        assert y.shape == (5,)
        assert y.requires_grad

    # test single sampled model
    sampled_net = bnet.sample_model()
    y = sampled_net(x)
    assert y.shape == (5,)
    assert y.requires_grad

    # Batched
    x = randn(256, 3)
    y = net(x)
    assert y.shape == (256, 5)

    # test context manager
    with bnet.sample() as sampled_net:
        y = sampled_net(x)
        assert y.shape == (256, 5)
        assert y.requires_grad

    # test single sampled model
    sampled_net = bnet.sample_model()
    y = sampled_net(x)
    assert y.shape == (256, 5)
    assert y.requires_grad

    # Jacobian
    x = randn(3)
    J = torch.autograd.functional.jacobian(net, x)
    assert (J[~adjacency] == 0).all()

    # test context manager
    with bnet.sample() as sampled_net:
        J = torch.autograd.functional.jacobian(sampled_net, x)
        assert (J[~adjacency] == 0).all()

    # test single sampled model
    sampled_net = bnet.sample_model()
    J = torch.autograd.functional.jacobian(sampled_net, x)
    assert (J[~adjacency] == 0).all()

    # compute KL divergence
    kl = bnet.kl_divergence()
    assert kl.item() >= 0


@pytest.mark.parametrize("F", [NICE, MAF, NSF, SOSPF, NAF, UNAF, GF, BPF])
def test_bayesian_flows(tmp_path: Path, F: callable):
    flow = F(3, 5)
    bflow = BayesianModel(flow)

    # Evaluation of log_prob
    x, c = randn(256, 3), randn(5)

    sampled_flow = bflow.sample_model()
    log_p = sampled_flow(c).log_prob(x)

    assert log_p.shape == (256,)
    assert log_p.requires_grad

    sampled_flow.zero_grad(set_to_none=True)
    loss = -log_p.mean()
    loss.backward()

    for p in sampled_flow.parameters():
        assert p.grad is not None

    # Sampling
    x = sampled_flow(c).sample((32,))

    assert x.shape == (32, 3)

    # Correct inverse transformation
    sampled_flow = bflow.sample_model()
    y = sampled_flow(c).transform(x)
    z = sampled_flow(c).transform.inv(y)
    assert torch.allclose(x, z, atol=1e-5)

    # Reparameterization trick
    if sampled_flow(c).has_rsample:
        x = sampled_flow(c).rsample()

        sampled_flow.zero_grad(set_to_none=True)
        loss = x.square().sum().sqrt()
        loss.backward()

        for p in sampled_flow.parameters():
            assert p.grad is not None

    # Testing the sampling of log_prob
    with bflow.sample() as sflow:
        log_p_i = sflow(c).log_prob(x)
        assert not torch.allclose(log_p_i, log_p)

    # Saving
    torch.save(bflow, tmp_path / "flow.pth")

    # Loading
    flow_bis = torch.load(tmp_path / "flow.pth")
    assert flow_bis
    # Printing
    assert repr(bflow)
