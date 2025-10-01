r"""Tests for the zuko.bayesian module."""

import math
import pytest
import torch

from pathlib import Path
from torch import randn
from typing import Sequence

from zuko.bayesian import BayesianModel
from zuko.flows import *
from zuko.nn import *


@pytest.mark.parametrize("local_trick", [False, True])
@pytest.mark.parametrize("batch", [(), (256,)])
def test_bayesian_MLP(local_trick: bool, batch: Sequence[int]):
    net = MLP(3, 5)
    bnet = BayesianModel(net)

    x = randn(*batch, 3)

    # reparameterize
    with bnet.reparameterize(local_trick=local_trick) as rnet:
        y1, y2 = rnet(x), rnet(x)

    with bnet.reparameterize(local_trick=local_trick) as rnet:
        y3 = rnet(x)

    assert y1.shape == (*batch, 5)
    assert y1.requires_grad
    assert torch.allclose(y1, y2)
    assert not torch.allclose(y1, y3)

    # sample_model
    snet = bnet.sample_model()
    y1, y2 = snet(x), snet(x)

    snet = bnet.sample_model()
    y3 = snet(x)

    assert y1.shape == (*batch, 5)
    assert y1.requires_grad
    assert torch.allclose(y1, y2)
    assert not torch.allclose(y1, y3)

    # kl_divergence
    kl = bnet.kl_divergence()

    assert torch.all(kl >= 0)


@pytest.mark.parametrize("local_trick", [False, True])
@pytest.mark.parametrize("batch", [(), (256,)])
def test_bayesian_MaskedMLP(local_trick: bool, batch: Sequence[int]):
    adjacency = randn(5, 3) < 0
    net = MaskedMLP(adjacency)
    bnet = BayesianModel(net)

    x = randn(*batch, 3)

    # reparameterize
    with bnet.reparameterize(local_trick=local_trick) as rnet:
        y1, y2 = rnet(x), rnet(x)

    with bnet.reparameterize(local_trick=local_trick) as rnet:
        y3 = rnet(x)

    assert y1.shape == (*batch, 5)
    assert y1.requires_grad
    assert torch.allclose(y1, y2)
    assert not torch.allclose(y1, y3)

    ## Jacobian
    with bnet.reparameterize() as rnet:
        x = randn(*batch, 3)
        J = torch.autograd.functional.jacobian(rnet, x)
        J = J.movedim(len(batch), -2)

    mask = torch.eye(math.prod(batch), dtype=bool)
    mask = mask.reshape(batch + batch)

    assert (J[mask][..., ~adjacency] == 0).all()
    assert (J[~mask] == 0).all()

    # sample_model
    snet = bnet.sample_model()
    y1, y2 = snet(x), snet(x)

    snet = bnet.sample_model()
    y3 = snet(x)

    assert y1.shape == (*batch, 5)
    assert y1.requires_grad
    assert torch.allclose(y1, y2)
    assert not torch.allclose(y1, y3)

    ## Jacobian
    snet = bnet.sample_model()
    x = randn(*batch, 3)
    J = torch.autograd.functional.jacobian(snet, x)
    J = J.movedim(len(batch), -2)

    mask = torch.eye(math.prod(batch), dtype=bool)
    mask = mask.reshape(batch + batch)

    assert (J[mask][..., ~adjacency] == 0).all()
    assert (J[~mask] == 0).all()


@pytest.mark.parametrize("F", [NICE, MAF, NSF, SOSPF, NAF, UNAF, GF, BPF])
@pytest.mark.parametrize("local_trick", [True, False])
def test_bayesian_flows(tmp_path: Path, F: type, local_trick: bool):
    flow = F(3, 5)
    bflow = BayesianModel(flow)

    # Evaluation of log_prob
    x, c = randn(256, 3), randn(5)

    ## reparametrize
    with bflow.reparameterize(local_trick=local_trick) as rflow:
        log_p1 = rflow(c).log_prob(x)
        log_p2 = rflow(c).log_prob(x)

    with bflow.reparameterize(local_trick=local_trick) as rflow:
        log_p3 = rflow(c).log_prob(x)

    assert log_p1.shape == (256,)
    assert log_p1.requires_grad
    assert torch.allclose(log_p1, log_p2)
    assert not torch.allclose(log_p1, log_p3)

    bflow.zero_grad(set_to_none=True)
    loss = -log_p3.mean()
    loss.backward()

    for p in bflow.parameters(recurse=False):
        assert p.grad is not None

    for p in bflow.base.parameters():
        assert p.grad is None

    ## sample_model
    sflow = bflow.sample_model()
    log_p1 = sflow(c).log_prob(x)
    log_p2 = sflow(c).log_prob(x)

    sflow = bflow.sample_model()
    log_p3 = sflow(c).log_prob(x)

    assert log_p1.shape == (256,)
    assert log_p1.requires_grad
    assert torch.allclose(log_p1, log_p2)
    assert not torch.allclose(log_p1, log_p3)

    bflow.zero_grad(set_to_none=True)
    sflow.zero_grad(set_to_none=True)
    loss = -log_p3.mean()
    loss.backward()

    for p in bflow.parameters():
        assert p.grad is None

    for p in sflow.parameters():
        assert p.grad is not None

    del sflow

    # Sampling
    with bflow.reparameterize(local_trick=local_trick) as rflow:
        x = rflow(c).sample((32,))

        assert x.shape == (32, 3)

    # Reparameterization trick
    if bflow.base(c).has_rsample:
        with bflow.reparameterize(local_trick=local_trick) as rflow:
            x = rflow(c).rsample()

        bflow.zero_grad(set_to_none=True)
        loss = x.square().sum().sqrt()
        loss.backward()

        for p in bflow.parameters(recurse=False):
            assert p.grad is not None

        for p in bflow.base.parameters():
            assert p.grad is None

    # Invertibility
    x, c = randn(256, 3), randn(256, 5)

    with bflow.reparameterize(local_trick=local_trick) as rflow:
        t = rflow(c).transform
        z = t.inv(t(x))

        assert torch.allclose(x, z, atol=1e-4)

    # Saving
    torch.save(bflow, tmp_path / "flow.pth")

    # Loading
    bflow_bis = torch.load(tmp_path / "flow.pth", weights_only=False)

    x, c = randn(3), randn(5)

    with torch.random.fork_rng():
        with bflow.reparameterize(local_trick=local_trick) as rflow:
            log_p = rflow(c).log_prob(x)

    with torch.random.fork_rng():
        with bflow_bis.reparameterize(local_trick=local_trick) as rflow:
            log_p_bis = rflow(c).log_prob(x)

    assert torch.allclose(log_p, log_p_bis)

    # Printing
    assert repr(bflow)
