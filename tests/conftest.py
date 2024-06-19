r"""Tests configuration."""

import pytest
import torch


@pytest.fixture(autouse=True, scope='module')
def torch_float64():
    try:
        yield torch.set_default_dtype(torch.float64)
    finally:
        torch.set_default_dtype(torch.float32)
