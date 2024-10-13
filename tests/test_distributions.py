r"""Tests for the zuko.distributions module."""

import torch

from torch.distributions import *

from zuko.distributions import *


def test_distributions():
    ds = [
        NormalizingFlow(ExpTransform(), Gamma(2.0, 1.0)),
        Joint(Uniform(0.0, 1.0), Normal(0.0, 1.0)),
        Mixture(Normal(torch.randn(2), torch.ones(2)), torch.randn(2)),
        GeneralizedNormal(2.0),
        DiagNormal(torch.zeros(2), torch.ones(2)),
        BoxUniform(-torch.ones(2), torch.ones(2)),
        TransformedUniform(ExpTransform(), -1.0, 1.0),
        Truncated(Normal(0.0, 1.0), 1.0, 2.0),
        Sort(Normal(0.0, 1.0), 2),
        TopK(Normal(0.0, 1.0), 2, 3),
        Minimum(Normal(0.0, 1.0), 3),
        Maximum(Normal(0.0, 1.0), 3),
    ]

    shape = (2**18,)

    for d in ds:
        assert d.batch_shape == (), d

        # Shapes
        x = d.sample(shape)

        assert x.shape == shape + d.event_shape, d

        log_p = d.log_prob(x)

        assert log_p.shape == shape, d

        # Expectation
        lower, upper = x.min(dim=0).values, x.max(dim=0).values
        width = upper - lower

        x = Uniform(lower - width / 2, upper + width / 2).sample(shape)

        p = d.log_prob(x).exp().mean() * (2 * width).prod()

        assert (0.9 <= p) and (p <= 1.1), d

        # Expand
        d = d.expand((32,))

        assert d.batch_shape == (32,), d

        x = d.sample()

        assert x.shape == d.batch_shape + d.event_shape, d

        log_p = d.log_prob(x)

        assert log_p.shape == d.batch_shape, d
