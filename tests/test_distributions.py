r"""Tests for the zuko.distributions module."""

import pytest
import torch

from torch.distributions import *

from zuko.distributions import *


@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3)])
def test_distributions(batch_shape: tuple[int, ...]) -> None:
    ds = [
        NormalizingFlow(ExpTransform(), Gamma(2.0 * torch.ones(batch_shape), 1.0)),
        Joint(Uniform(torch.zeros(batch_shape), 1.0), Normal(0.0, 1.0)),
        Mixture(Normal(torch.randn(*batch_shape, 2), 1.0), torch.randn(*batch_shape, 2)),
        GeneralizedNormal(2.0 * torch.ones(batch_shape)),
        DiagNormal(torch.zeros(*batch_shape, 2), torch.ones(*batch_shape, 2)),
        BoxUniform(-torch.ones(*batch_shape, 2), torch.ones(*batch_shape, 2)),
        TransformedUniform(ExpTransform(), -torch.ones(batch_shape), 1.0),
        Truncated(Normal(torch.zeros(batch_shape), 1.0), 1.0, 2.0),
        Truncated(Normal(0.0, 1.0), -torch.ones(batch_shape), torch.ones(batch_shape)),
        Joint(Truncated(Normal(0.0, 1.0), -torch.ones(batch_shape), 1.0), Normal(0.0, 1.0)),
        Sort(Normal(torch.zeros(batch_shape), 1.0), 2),
        TopK(Normal(torch.zeros(batch_shape), 1.0), 2, 3),
        Minimum(Normal(torch.zeros(batch_shape), 1.0), 3),
        Maximum(Normal(torch.zeros(batch_shape), 1.0), 3),
    ]

    shape = (2**16,)

    for d in ds:
        assert d.batch_shape == batch_shape, d

        # Shapes
        x = d.sample(shape)

        assert x.shape == shape + d.batch_shape + d.event_shape, d

        log_p = d.log_prob(x)

        assert log_p.shape == shape + d.batch_shape, d

        # Expectation
        lower, upper = x.min(dim=0).values, x.max(dim=0).values
        width = upper - lower

        x = Uniform(lower - width / 2, upper + width / 2).sample(shape)

        p = d.log_prob(x).exp().mean(dim=0)
        p = p * (2 * width).reshape(*batch_shape, -1).prod(dim=-1)

        assert torch.all((0.9 <= p) & (p <= 1.1)), d

        # Expand
        d = d.expand((32, *batch_shape))

        assert d.batch_shape == (32, *batch_shape), d

        x = d.sample()

        assert x.shape == d.batch_shape + d.event_shape, d

        log_p = d.log_prob(x)

        assert log_p.shape == d.batch_shape, d
