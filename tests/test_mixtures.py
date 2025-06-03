r"""Tests for the zuko.mixtures module."""

import pytest
import torch

from pathlib import Path
from torch import randn
from typing import Sequence

from zuko.mixtures import *


@pytest.mark.parametrize("M", [GMM])
def test_mixtures(tmp_path: Path, M: callable):
    mixture = M(3, 5)

    # Evaluation of log_prob
    x, c = randn(256, 3), randn(5)
    log_p = mixture(c).log_prob(x)

    assert log_p.shape == (256,)
    assert log_p.requires_grad

    mixture.zero_grad(set_to_none=True)
    loss = -log_p.mean()
    loss.backward()

    for p in mixture.parameters():
        assert p.grad is not None

    # Sampling
    x = mixture(c).sample((32,))

    assert x.shape == (32, 3)

    # Reparameterization trick
    if mixture(c).has_rsample:
        x = mixture(c).rsample()

        mixture.zero_grad(set_to_none=True)
        loss = x.square().sum().sqrt()
        loss.backward()

        for p in mixture.parameters():
            assert p.grad is not None

    # Saving
    torch.save(mixture, tmp_path / "mixture.pth")

    # Loading
    mixture_bis = torch.load(tmp_path / "mixture.pth", weights_only=False)

    x, c = randn(3), randn(5)

    seed = torch.seed()
    log_p = mixture(c).log_prob(x)
    torch.manual_seed(seed)
    log_p_bis = mixture_bis(c).log_prob(x)

    assert torch.allclose(log_p, log_p_bis)

    # Printing
    assert repr(mixture)


@pytest.mark.parametrize("batch", [(), (4,)])
@pytest.mark.parametrize("features", [3])
@pytest.mark.parametrize("context", [0, 5])
@pytest.mark.parametrize("components", [2])
@pytest.mark.parametrize("covariance_type", ["full", "diagonal", "spherical"])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("strategy", [None, "random", "kmeans", "kmeans++"])
def test_gmm_shapes(
    batch: Sequence[int],
    features: int,
    context: int,
    components: int,
    covariance_type: str,
    tied: bool,
    strategy: str,
):
    if context > 0:
        c = torch.randn(*batch, context)
    else:
        c, batch = None, ()

    gmm = GMM(
        features=features,
        context=context,
        components=components,
        covariance_type=covariance_type,
        tied=tied,
    )

    # Initialization
    if strategy is not None:
        gmm.initialize(torch.randn(1024, features), strategy=strategy)

    # Forward
    d = gmm(c)

    # Shapes
    assert d.batch_shape == (*batch,)
    assert d.event_shape == (features,)

    assert d.logits.shape == (*batch, components)

    if covariance_type == "full":
        assert d.base.loc.shape == (*batch, components, features)
        assert d.base.covariance_matrix.shape == (*batch, components, features, features)
    else:
        assert d.base.base_dist.loc.shape == (*batch, components, features)
        assert d.base.base_dist.scale.shape == (*batch, components, features)


@pytest.mark.parametrize("covariance_type", ["full", "diagonal", "spherical"])
def test_gmm_tied_covariance(covariance_type: str):
    gmm = GMM(features=3, components=2, covariance_type=covariance_type, tied=True)
    d = gmm()

    if covariance_type == "full":
        covs = d.base.covariance_matrix
    else:
        covs = d.base.base_dist.scale

    assert torch.allclose(covs[0], covs[1])


def test_gmm_insufficient_samples():
    gmm = GMM(features=3, components=7, covariance_type="full")

    with pytest.raises(AssertionError, match="The number of samples"):
        gmm.initialize(torch.randn(6, 3), strategy="random")
