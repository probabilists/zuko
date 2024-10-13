r"""Doctests configuration."""

import pytest
import torch

import zuko


@pytest.fixture(autouse=True, scope="module")
def doctest_imports(doctest_namespace):
    doctest_namespace["torch"] = torch
    doctest_namespace["zuko"] = zuko


@pytest.fixture(autouse=True)
def torch_seed():
    with torch.random.fork_rng():
        yield torch.random.manual_seed(0)
