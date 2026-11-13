r"""Tests for the zuko.transforms module."""

import pytest
import torch

from torch import randn
from torch.distributions import *

from zuko.transforms import *


@pytest.mark.parametrize("batched", [False, True])
def test_univariate_transforms(batched: bool) -> None:
    batch = (256,) if batched else ()

    ts = [
        IdentityTransform(),
        CosTransform(),
        SinTransform(),
        SoftclipTransform(),
        CircularShiftTransform(),
        SignedPowerTransform(randn(batch)),
        AdditiveTransform(randn(batch)),
        MonotonicAffineTransform(randn(batch), randn(batch)),
        MonotonicRQSTransform(randn(*batch, 8), randn(*batch, 8), randn(*batch, 7)),
        MonotonicTransform(lambda x: x**3),
        BernsteinTransform(randn(*batch, 16)),
        BoundedBernsteinTransform(randn(*batch, 16)),
        GaussianizationTransform(randn(*batch, 8), randn(*batch, 8)),
        UnconstrainedMonotonicTransform(lambda x: torch.exp(-(x**2)) + 1e-2),
        SOSPolynomialTransform(randn(*batch, 3, 5)),
    ]

    for t in ts:
        # Call
        if hasattr(t.domain, "lower_bound"):
            x = torch.linspace(t.domain.lower_bound + 1e-2, t.domain.upper_bound - 1e-2, 256)
        else:
            x = torch.linspace(-5.0, 5.0, 256)

        y = t(x)

        assert x.shape == y.shape, t

        # Inverse
        z = t.inv(y)

        assert torch.allclose(x, z, atol=1e-4), t

        # Jacobian
        J = torch.autograd.functional.jacobian(t, x)

        assert (torch.triu(J, diagonal=+1) == 0).all(), t
        assert (torch.tril(J, diagonal=-1) == 0).all(), t

        ladj = torch.diag(J).abs().log()

        assert torch.allclose(t.log_abs_det_jacobian(x, y), ladj, atol=1e-4), t

        # Compound call
        y_comp, ladj_comp = t.call_and_ladj(x)

        assert torch.allclose(y_comp, y, atol=1e-4), t
        assert torch.allclose(ladj_comp, ladj, atol=1e-4), t

        # Inverse Jacobian
        J = torch.autograd.functional.jacobian(t.inv, y)

        assert (torch.triu(J, diagonal=+1) == 0).all(), t
        assert (torch.tril(J, diagonal=-1) == 0).all(), t

        ladj = torch.diag(J).abs().log()

        assert torch.allclose(t.inv.log_abs_det_jacobian(y, z), ladj, atol=1e-4), t


def test_multivariate_transforms() -> None:
    A, B = torch.randn(5, 16), torch.randn(16, 5)
    f = lambda t, x: torch.sigmoid(x @ A) @ B

    ts = [
        FreeFormJacobianTransform(f, 0.0, 1.0, atol=1e-7, rtol=1e-6),
        PermutationTransform(torch.randperm(5)),
        RotationTransform(randn(5, 5)),
        LULinearTransform(randn(5, 5)),
        PlanarTransform(5 * randn(5), randn(5), randn(())),
        PlanarTransform(randn(5), torch.zeros(5), randn(())),
        PlanarTransform(randn(5), 1e-8 * randn(5), randn(())),
        SylvesterTransform(torch.linalg.qr(randn(5, 3)).Q, randn(3, 3), randn(3, 3), randn(3)),
        SylvesterTransform(
            torch.linalg.qr(randn(5, 5)).Q, 3 * randn(5, 5), 3 * randn(5, 5), randn(5)
        ),
        TriangularSylvesterTransform(randn(5, 5), randn(5, 5), randn(5)),
        TriangularSylvesterTransform(randn(5, 5), randn(5, 5), randn(5), torch.randperm(5)),
    ]

    for t in ts:
        # Shapes
        x = randn(256, 5)
        y = t(x)

        assert x.shape == y.shape, t

        # Inverse
        z = t.inv(y)

        assert torch.allclose(x, z, atol=1e-4), t

        # Jacobian
        x = randn(5)
        y = t(x)

        J = torch.autograd.functional.jacobian(t, x)
        ladj = torch.linalg.slogdet(J).logabsdet

        assert torch.allclose(t.log_abs_det_jacobian(x, y), ladj, atol=1e-4), t

        # Compound call
        y_comp, ladj_comp = t.call_and_ladj(x)

        assert torch.allclose(y_comp, y, atol=1e-4), t
        assert torch.allclose(ladj_comp, ladj, atol=1e-4), t

        # Inverse Jacobian
        z = t.inv(y)

        J = torch.autograd.functional.jacobian(t.inv, y)
        ladj = torch.linalg.slogdet(J).logabsdet

        assert torch.allclose(t.inv.log_abs_det_jacobian(y, z), ladj, atol=1e-4), t


def test_saturated_tanh_residual_transforms() -> None:
    # The diagonal elements of C are -1 + d with d below the float32 resolution at -1
    w = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    R1 = torch.diag(torch.tensor([12.0, 1.0, 1.0]))
    R2 = torch.diag(torch.tensor([-12.0, 1.0, 1.0]))
    order = torch.randperm(3)

    ts = [
        lambda dtype: PlanarTransform(
            -20 * w.to(dtype), w.to(dtype), torch.zeros((), dtype=dtype)
        ),
        lambda dtype: SylvesterTransform(
            torch.eye(3, dtype=dtype), R1.to(dtype), R2.to(dtype), torch.zeros(3, dtype=dtype)
        ),
        lambda dtype: TriangularSylvesterTransform(
            R1.to(dtype), R2.to(dtype), torch.zeros(3, dtype=dtype), order
        ),
    ]

    for T in ts:
        t, t64 = T(torch.float32), T(torch.float64)

        x = torch.cat((torch.zeros(1, 3), 1e-3 * randn(15, 3), randn(16, 3))).float()
        y = t(x)

        J = torch.stack([torch.autograd.functional.jacobian(t64, xi) for xi in x.double()])
        ladj = torch.linalg.slogdet(J).logabsdet

        assert torch.allclose(t.log_abs_det_jacobian(x, y).double(), ladj, atol=1e-3), t
        assert torch.allclose(t.inv(y), x, atol=1e-3), t


def test_planar_transform_small_weights() -> None:
    u, b = randn(5, dtype=torch.float64), randn((), dtype=torch.float64)
    w = randn(5, dtype=torch.float64)
    w = w / torch.linalg.vector_norm(w)

    for scale in [1e-2, 1e-4, 1e-6, 1e-8, 0.0]:
        t64 = PlanarTransform(u, scale * w, b)
        t = PlanarTransform(u.float(), scale * w.float(), b.float())

        assert torch.linalg.vector_norm(t.A) <= torch.linalg.vector_norm(u) + 1e-6, scale

        x = randn(256, 5, dtype=torch.float64)
        y = t64(x)

        assert torch.allclose(t.inv(y.float()).double(), x, atol=1e-5), scale

        J = torch.autograd.functional.jacobian(t64, x[0])
        ladj = torch.linalg.slogdet(J).logabsdet

        assert torch.allclose(
            t.log_abs_det_jacobian(x[:1].float(), y[:1].float()).double(), ladj, atol=1e-5
        ), scale

    # Gradients at w = 0
    u, w = randn(5, requires_grad=True), torch.zeros(5, requires_grad=True)
    t = PlanarTransform(u, w, randn(()))
    x = randn(5)
    (t(x).sum() + t.log_abs_det_jacobian(x, t(x)) + t.inv(x).sum()).backward()

    assert torch.isfinite(u.grad).all() and torch.isfinite(w.grad).all()


@pytest.mark.parametrize(
    ("batch_shape", "order_shape"),
    [
        ((), ()),
        ((4,), ()),
        ((), (2,)),
        ((2,), (2,)),
        ((4, 1, 2), (3, 2)),
        ((4, 3, 2), (1, 2)),
        ((4, 3, 2), ()),
    ],
)
def test_permutation_transform_broadcast(
    batch_shape: tuple[int, ...], order_shape: tuple[int, ...]
) -> None:
    orders = torch.tensor([[2, 0, 1], [1, 2, 0], [0, 2, 1]])
    indices = torch.arange(torch.Size(order_shape).numel()) % len(orders)
    order = orders[indices].reshape(*order_shape, 3)
    x = torch.arange(torch.Size(batch_shape).numel() * 3).to(torch.get_default_dtype())
    x = x.reshape(*batch_shape, 3).requires_grad_()
    shape = torch.broadcast_shapes(batch_shape, order_shape) + (3,)

    # Apply each permutation independently as a reference.
    expected = torch.stack([
        xi[oi]
        for xi, oi in zip(
            x.expand(shape).reshape(-1, 3), order.expand(shape).reshape(-1, 3), strict=True
        )
    ]).reshape(shape)

    t = PermutationTransform(order)
    y, ladj = t.call_and_ladj(x)

    assert y.shape == shape
    assert torch.equal(y, expected)
    assert ladj.shape == shape[:-1]
    assert torch.equal(ladj, torch.zeros(shape[:-1]))

    z, inverse_ladj = t.inv.call_and_ladj(y)

    assert torch.equal(z, x.expand(shape))
    assert inverse_ladj.shape == shape[:-1]
    assert torch.equal(inverse_ladj, -ladj)

    weights = torch.arange(y.numel()).reshape(y.shape).to(y)
    actual_grad = torch.autograd.grad((y * weights).sum(), x)[0]
    expected_grad = torch.autograd.grad((expected * weights).sum(), x)[0]

    assert torch.equal(actual_grad, expected_grad)


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
@pytest.mark.parametrize("batched", [False, True])
def test_permutation_transform_cpu_order(device: str, batched: bool) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    order = torch.tensor([[2, 0, 1], [1, 2, 0]])
    if not batched:
        order = order[0]
    original = order.clone()
    x = torch.arange(24, dtype=torch.float32, device=device).reshape(4, 2, 3)
    x.requires_grad_()
    permutation = torch.eye(3, dtype=x.dtype)[order].to(device)
    expected = (permutation @ x.unsqueeze(-1)).squeeze(-1)
    t = PermutationTransform(order)

    y, ladj = t.call_and_ladj(x)
    recovered, inverse_ladj = t.inv.call_and_ladj(y)

    torch.testing.assert_close(y, expected)
    torch.testing.assert_close(recovered, x)
    assert y.device == x.device and y.dtype == x.dtype
    assert ladj.shape == x.shape[:-1]
    torch.testing.assert_close(ladj, torch.zeros_like(x[..., 0]))
    torch.testing.assert_close(inverse_ladj, -ladj)
    assert t.order is order
    assert t.order.device.type == "cpu" and t.order.dtype == torch.long
    assert torch.equal(t.order, original)

    weights = torch.arange(y.numel(), device=device).reshape(y.shape).to(y)
    actual_grad = torch.autograd.grad((y * weights).sum(), x)[0]
    expected_grad = torch.autograd.grad((expected * weights).sum(), x)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
