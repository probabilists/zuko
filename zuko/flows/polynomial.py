r"""Polynomial flows."""

__all__ = [
    "BPF",
    "SOSPF",
]

from .autoregressive import MAF
from ..lazy import UnconditionalTransform
from ..transforms import BoundedBernsteinTransform, SoftclipTransform, SOSPolynomialTransform


class SOSPF(MAF):
    r"""Creates a sum-of-squares polynomial flow (SOSPF).

    Warning:
        Invertibility is only guaranteed for features within the interval :math:`[-10,
        10]`. It is recommended to standardize features (zero mean, unit variance)
        before training.

    See also:
        :class:`zuko.transforms.SOSPolynomialTransform`

    References:
        | Sum-of-Squares Polynomial Flow (Jaini et al., 2019)
        | https://arxiv.org/abs/1905.02325

    Arguments:
        features: The number of features.
        context: The number of context features.
        degree: The degree :math:`L` of polynomials.
        polynomials: The number of polynomials :math:`K`.
        kwargs: Keyword arguments passed to :class:`zuko.flows.autoregressive.MAF`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        degree: int = 4,
        polynomials: int = 3,
        **kwargs,
    ):
        super().__init__(
            features=features,
            context=context,
            univariate=SOSPolynomialTransform,
            shapes=[(polynomials, degree + 1), ()],
            **kwargs,
        )

        transforms = self.transform.transforms

        for i in reversed(range(1, len(transforms))):
            transforms.insert(i, UnconditionalTransform(SoftclipTransform, bound=11.0))


class BPF(MAF):
    r"""Creates a Bernstein polynomial flow (BPF).

    Warning:
        The Bernstein polynomial is bounded to the interval :math:`[-5, 5]`. Any feature
        outside of this domain is not transformed. It is recommended to standardize
        features (zero mean, unit variance) before training.

    See also:
        :class:`zuko.transforms.BoundedBernsteinTransform`

    References:
        | Deep transformation models: Tackling complex regression problems with neural network based transformation models (Sick et al., 2020)
        | https://arxiv.org/abs/2004.00464

        | Short-Term Density Forecasting of Low-Voltage Load using Bernstein-Polynomial Normalizing Flows (Arpogaus et al., 2022)
        | https://arxiv.org/abs/2204.13939

    Arguments:
        features: The number of features.
        context: The number of context features.
        degree: The degree :math:`M` of the Bernstein polynomial.
        kwargs: Keyword arguments passed to :class:`zuko.flows.autoregressive.MAF`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        degree: int = 16,
        **kwargs,
    ):
        super().__init__(
            features=features,
            context=context,
            univariate=BoundedBernsteinTransform,
            shapes=[(degree + 1,)],
            **kwargs,
        )
