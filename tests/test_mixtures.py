r"""Tests for the zuko.mixtures module."""

import pytest
import torch

from pathlib import Path
from torch import randn

from zuko.mixtures import *


@pytest.mark.parametrize("M", [GMM])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mixtures(tmp_path: Path, M: callable, device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("No CUDA devices available")

    torch.set_default_device(device)

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
    mixture_bis = torch.load(tmp_path / "mixture.pth")

    x, c = randn(3), randn(5)

    seed = torch.seed()
    log_p = mixture(c).log_prob(x)
    torch.manual_seed(seed)
    log_p_bis = mixture_bis(c).log_prob(x)

    assert torch.allclose(log_p, log_p_bis)

    # Printing
    assert repr(mixture)
