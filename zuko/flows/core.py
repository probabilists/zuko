r"""Core building blocks.

Warning:
    This sub-module is deprecated and will be removed in the future. Use
    :mod:`zuko.lazy` instead.
"""

__all__ = [
    "Flow",
]

from ..lazy import (  # noqa: F401
    Flow,
    LazyComposedTransform,
    LazyDistribution,
    LazyInverse,
    LazyTransform,
    Unconditional,
    UnconditionalDistribution,
    UnconditionalTransform,
)
