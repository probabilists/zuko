r"""Tests for the zuko.distributions module."""

import pytest
import torch

from torch.distributions import *

from zuko.distributions import *


def test_distributions() -> None:
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


@pytest.mark.parametrize(
    "base_shape,lower_shape,upper_shape,batch_shape",
    [
        ((), (), (), ()),
        ((), (3,), (), (3,)),
        ((), (), (3,), (3,)),
        ((), (2, 1), (3,), (2, 3)),
        ((2, 1), (), (3,), (2, 3)),
        ((2, 3), (), (), (2, 3)),
        ((2, 1), (1, 3), (), (2, 3)),
    ],
)
def test_truncated_batch_shape(
    base_shape: tuple[int, ...],
    lower_shape: tuple[int, ...],
    upper_shape: tuple[int, ...],
    batch_shape: tuple[int, ...],
) -> None:
    base = Normal(torch.zeros(base_shape), torch.ones(base_shape))
    d = Truncated(base, -torch.ones(lower_shape), torch.ones(upper_shape))

    assert d.batch_shape == batch_shape

    for distribution in (d, d.expand((4,) + batch_shape)):
        x = distribution.rsample((5,))

        assert x.shape == (5,) + distribution.batch_shape
        assert distribution.log_prob(x).shape == x.shape
        assert distribution.cdf(x).shape == x.shape
        assert torch.all((-1 <= x) & (x < 1))


def test_truncated_joint() -> None:
    base = Normal(0.0, 1.0)
    truncated = Truncated(base, torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 2.0]))
    joint = Joint(truncated, base)
    x = joint.rsample((5,))

    assert joint.batch_shape == (2,)
    assert x.shape == (5, 2, 2)
    assert torch.allclose(
        joint.log_prob(x), truncated.log_prob(x[..., 0]) + base.log_prob(x[..., 1])
    )


def test_truncated_mixture() -> None:
    base = Normal(0.0, 1.0)
    truncated = Truncated(base, torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 2.0]))
    mixture = Mixture(truncated, torch.zeros(2))
    x = mixture.sample((5,))
    expected = torch.logsumexp(truncated.log_prob(x.unsqueeze(-1)), dim=-1) - torch.log(
        torch.tensor(2.0)
    )

    assert mixture.batch_shape == ()
    assert x.shape == (5,)
    assert torch.allclose(mixture.log_prob(x), expected)
