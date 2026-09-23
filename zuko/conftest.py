r"""Doctests configuration."""

import pytest
import torch

from collections.abc import Iterator

import zuko


@pytest.fixture(autouse=True, scope="module")
def doctest_imports(doctest_namespace: dict) -> None:
    doctest_namespace["torch"] = torch
    doctest_namespace["zuko"] = zuko


@pytest.fixture(autouse=True)
def torch_seed() -> Iterator[None]:
    with torch.random.fork_rng():
        yield torch.random.manual_seed(0)
