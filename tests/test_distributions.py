r"""Tests for the zuko.distributions module."""

import pytest
import torch

from torch.distributions import *

from zuko.distributions import *
from zuko.transforms import PermutationTransform


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


@pytest.mark.parametrize("sample_shape", [(), (4,), (4, 5)])
def test_normalizing_flow_batched_permutation(sample_shape: tuple[int, ...]) -> None:
    order = torch.tensor([[2, 0, 1], [1, 2, 0]])
    loc = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    scale = torch.tensor([[1.0, 2.0, 4.0], [0.5, 1.0, 2.0]])
    flow = NormalizingFlow(PermutationTransform(order), DiagNormal(loc, scale))
    inverse = torch.argsort(order)
    reference = DiagNormal(
        torch.stack([loc[i, inverse[i]] for i in range(2)]),
        torch.stack([scale[i, inverse[i]] for i in range(2)]),
    )

    x, log_p = flow.rsample_and_log_prob(sample_shape)

    assert x.shape == sample_shape + flow.batch_shape + flow.event_shape
    assert log_p.shape == sample_shape + flow.batch_shape
    assert torch.allclose(log_p, reference.log_prob(x))
    assert torch.allclose(log_p, flow.log_prob(x))

    with torch.random.fork_rng():
        z = flow.base.rsample(sample_shape)
        expected = torch.stack([z[..., i, inverse[i]] for i in range(2)], dim=-2)

    assert torch.equal(flow.rsample(sample_shape), expected)

    grad = torch.autograd.grad(x.sum(), loc)[0]

    assert torch.equal(grad, torch.full_like(loc, torch.Size(sample_shape).numel()))


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
@pytest.mark.parametrize("batched", [False, True])
def test_normalizing_flow_cpu_permutation_order(device: str, batched: bool) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    order = torch.tensor([[2, 0, 1], [1, 2, 0]])
    if not batched:
        order = order[0]
    original = order.clone()
    loc = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32, device=device, requires_grad=True
    )
    scale = torch.tensor([[1.0, 2.0, 4.0], [0.5, 1.0, 2.0]], dtype=loc.dtype, device=device)
    t = PermutationTransform(order)
    flow = NormalizingFlow(t, DiagNormal(loc, scale))
    inverse = torch.eye(3, dtype=loc.dtype)[order].to(device).transpose(-1, -2)
    reference = DiagNormal(
        (inverse @ loc.unsqueeze(-1)).squeeze(-1),
        (inverse @ scale.unsqueeze(-1)).squeeze(-1),
    )

    x, log_p = flow.rsample_and_log_prob((4,))

    assert x.shape == (4, 2, 3) and x.device == loc.device and x.dtype == loc.dtype
    assert log_p.shape == (4, 2)
    torch.testing.assert_close(log_p, reference.log_prob(x))
    torch.testing.assert_close(log_p, flow.log_prob(x))
    weights = torch.arange(x.numel(), device=device).reshape(x.shape).to(x)
    actual_grad = torch.autograd.grad((x * weights).sum(), loc)[0]
    expected_grad = (inverse.transpose(-1, -2) @ weights.unsqueeze(-1)).squeeze(-1).sum(0)
    torch.testing.assert_close(actual_grad, expected_grad)
    assert t.order is order
    assert t.order.device.type == "cpu" and t.order.dtype == torch.long
    assert torch.equal(t.order, original)
