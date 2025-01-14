import pytest
import torch

from zuko.mixtures import GMM


@pytest.fixture
def data_3d():
    torch.manual_seed(0)
    return torch.randn(100, 3)


def check_gmm(gmm, out_dist, means_shape, covariances_shape, weights_shape):
    if gmm.covariance_type == "full":
        means = out_dist.base.loc
        covariances = out_dist.base.covariance_matrix
    else:
        means = out_dist.base.base_dist.loc
        covariances = out_dist.base.base_dist.scale
    weights = out_dist.logits

    assert means.shape == means_shape
    assert covariances.shape == covariances_shape
    assert weights.shape == weights_shape


@pytest.mark.parametrize(
    "covariance_type, means_shape, covariances_shape",
    [
        ("full", (2, 3), (2, 3, 3)),
        ("diag", (2, 3), (2, 3)),
        ("spherical", (2, 3), (2, 3)),  # spherical is broadcasted to diagonal
    ],
)
def test_shape_unconditional(covariance_type, means_shape, covariances_shape):
    components = 2
    features = 3

    gmm = GMM(features=features, components=components, covariance_type=covariance_type)
    check_gmm(gmm, gmm(), means_shape, covariances_shape, (2,))

    dist = gmm()

    assert dist.batch_shape == torch.Size([])
    assert dist.event_shape == (features,)


@pytest.mark.parametrize(
    "covariance_type, means_shape, covariances_shape",
    [
        ("full", (4, 2, 3), (4, 2, 3, 3)),
        ("diag", (4, 2, 3), (4, 2, 3)),
        ("spherical", (4, 2, 3), (4, 2, 3)),  # spherical is broadcasted to diagonal
    ],
)
def test_shape_conditional(covariance_type, means_shape, covariances_shape):
    components = 2
    features = 3
    context = 3
    context_batch = 4

    gmm = GMM(
        features=features,
        components=components,
        context=context,
        covariance_type=covariance_type,
    )

    context = torch.randn(context_batch, context)
    check_gmm(
        gmm,
        gmm(context),
        means_shape,
        covariances_shape,
        (
            4,
            2,
        ),
    )

    dist = gmm(context)

    assert dist.batch_shape == torch.Size([context_batch])
    assert dist.event_shape == (features,)


@pytest.mark.parametrize("covariance_type", ["full", "diag", "spherical"])
def test_tied_covariance(covariance_type):
    gmm = GMM(features=3, components=2, covariance_type=covariance_type, tied=True)

    out_dist = gmm()

    if covariance_type == "full":
        covariances = out_dist.base.covariance_matrix
    else:
        covariances = out_dist.base.base_dist.scale

    assert torch.all(covariances[0] == covariances[1])


@pytest.mark.parametrize("strategy", ["random", "kmeans", "kmeans++"])
def test_initialize_valid_strategies(strategy):
    features = 2
    components = 3
    samples = 128
    torch.manual_seed(0)
    f = torch.randn(samples, features)

    gmm = GMM(features=features, components=components, covariance_type="full")
    gmm.initialize(f, strategy=strategy)

    assert gmm.phi is not None
    assert len(gmm.phi) == 4  # weights, means, diag, off_diag
    assert gmm.phi[0].shape == (components,)
    assert gmm.phi[1].shape == (components, features)
    assert gmm.phi[2].shape == (components, features)
    assert gmm.phi[3].shape == (components, features * (features - 1) // 2)


def test_initialize_invalid_strategy():
    features = 2
    components = 3
    samples = 128
    torch.manual_seed(0)
    f = torch.randn(samples, features)

    gmm = GMM(features=features, components=components, covariance_type="full")

    with pytest.raises(ValueError, match="Invalid initialization strategy"):
        gmm.initialize(f, strategy="invalid_strategy")


def test_initialize_insufficient_samples():
    features = 2
    components = 3
    samples = 2
    torch.manual_seed(0)
    f = torch.randn(samples, features)

    gmm = GMM(features=features, components=components, covariance_type="full")

    with pytest.raises(ValueError, match="Number of samples"):
        gmm.initialize(f, strategy="random")
