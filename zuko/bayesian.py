r"""Utils for bayesian flows."""

from __future__ import annotations

__all__ = [
    "BayesianModel",
]

import copy
import math
import re
import torch
import torch.nn as nn

from contextlib import contextmanager
from functools import partial
from torch import Tensor
from typing import Dict, Generator, Sequence

from .nn import Linear, MaskedLinear, linear


def _compile(prefix: str) -> re.Pattern:
    assert re.fullmatch(r"[\w\.\*]*", prefix) is not None, f"Invalid prefix {prefix}."

    # fmt: off
    pattern = (
        prefix
        .replace(".", r"\.")
        .replace("**", r"[\w\.]+")
        .replace("*", r"\w+")
        + r".*"
    )
    # fmt: on

    return re.compile(pattern)


def _match(pattern: re.Pattern, string: str) -> bool:
    return re.fullmatch(pattern, string) is not None


def _softclip(x: Tensor, bound: float) -> Tensor:
    return x * (1 + (x / bound).square()).rsqrt()


class BayesianModel(nn.Module):
    r"""Creates a Bayesian wrapper around a base model.

    Arguments:
        base: A base model.
        include_params: A list of parameter name prefixes to include. In a prefix, a
            single star `*` matches alphanumeric strings (`[a-zA-Z0-9_]+`) and a double
            star `**` matches dot-separated alphanumeric strings (`[a-zA-Z0-9_\.]+`). By
            default, all parameters are included.
        exclude_params: A list of parameter name prefixes to exclude.
    """

    def __init__(
        self,
        base: nn.Module,
        init_logvar: float = -9.0,
        include_params: Sequence[str] = ("",),
        exclude_params: Sequence[str] = (),
    ):
        super().__init__()

        self.base = base

        self.means = nn.ParameterDict()
        self.logvars = nn.ParameterDict()

        include_patterns = [_compile(prefix) for prefix in include_params]
        exclude_patterns = [_compile(prefix) for prefix in exclude_params]

        for name, param in self.base.named_parameters():
            if not any(_match(pattern, name) for pattern in include_patterns):
                continue
            elif any(_match(pattern, name) for pattern in exclude_patterns):
                continue

            key = name.replace(".", "-")

            self.means[key] = nn.Parameter(torch.empty_like(param))
            self.logvars[key] = nn.Parameter(torch.empty_like(param))

        self._reset_bayesian_parameters(logvar_mean=init_logvar)

    def _reset_bayesian_parameters(self, logvar_mean: float = -9.0, logvar_std: float = 1e-3):
        r"""Initializes the posterior means and log-variances."""

        for name, param in self.base.named_parameters():
            key = name.replace(".", "-")

            if key in self.means:
                self.means[key].data.copy_(param.data)
                self.logvars[key].data.normal_(logvar_mean, logvar_std)

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "BayesianModel should not be called directly. Use `sample_model` or `reparameterize` instead."
        )

    def sample_params(self) -> Dict[str, Tensor]:
        r"""Returns model parameters sampled from the posterior."""

        params = {}

        for key in self.means.keys():
            name = key.replace("-", ".")

            mean = self.means[key]
            std = _softclip(self.logvars[key], 11.0).div(2.0).exp()

            params[name] = mean + std * torch.randn_like(mean)

        return params

    def sample_model(self) -> nn.Module:
        r"""Returns a model sampled from the posterior.

        The returned model is a standalone copy of the base model where parameters have
        been replaced. It can be used exactly in the same ways as the base model.

        Warning:
            This method should not be used for training as it cannot propagate gradients
            to the Bayesian model's internal parameters.
        """

        with torch.no_grad():
            params = self.sample_params()

        new_base = copy.deepcopy(self.base)
        new_base.load_state_dict(params, strict=False)

        return new_base

    @contextmanager
    def reparameterize(self, local_trick: bool = False) -> Generator[nn.Module]:
        r"""Reparameterizes the base model from the posterior.

        Within this context, the base model is temporarily reparametrized and behaves
        deterministically, meaning that two calls to a method with the same inputs,
        leads to the same outputs.

        Arguments:
            local_trick: Whether to use the local reparameterization trick for linear
                layers.

        Yields:
            The reparametrized base model.

        References:
            | Variational Dropout and the Local Reparameterization Trick (Kingma et al., 2015)
            | https://arxiv.org/abs/1506.02557
        """

        params = self.sample_params()

        with torch.nn.utils.stateless._reparametrize_module(self.base, params):
            if local_trick:
                with self._reparametrize_trick():
                    yield self.base
            else:
                yield self.base

    @contextmanager
    def _reparametrize_trick(self):
        original_forwards = {}
        randn_cache = {}

        for name, module in self.base.named_modules():
            key = name.replace(".", "-")

            if type(module) not in (nn.Linear, Linear, MaskedLinear):
                continue
            elif key + "-weight" not in self.means:
                continue

            original_forwards[name] = module.forward
            module.forward = partial(
                self._local_reparameterization_trick,
                name=name,
                module=module,
                randn_cache=randn_cache,
            )

        try:
            yield
        finally:
            for name, module in self.base.named_modules():
                if name in original_forwards:
                    module.forward = original_forwards[name]

    def _local_reparameterization_trick(
        self,
        x: Tensor,
        name: str,
        module: nn.Module,
        randn_cache: Dict[(str, torch.Size), Tensor],
    ) -> Tensor:
        """Applies the local reparameterization trick to a linear layer."""

        key = name.replace(".", "-")

        w_mean = self.means[key + "-weight"]
        w_var = _softclip(self.logvars[key + "-weight"], 11.0).exp()

        if isinstance(module, MaskedLinear):
            w_mean = module.mask * w_mean
            w_var = module.mask * w_var

        if key + "-bias" in self.means:
            b_mean = self.means[key + "-bias"]
            b_var = _softclip(self.logvars[key + "-bias"], 11.0).exp()
        else:
            b_mean = module.bias
            b_var = None

        y_mean = linear(x, w_mean, b_mean)
        y_var = linear(x.square(), w_var, b_var)

        address = (name, y_mean.shape)

        if address in randn_cache:
            eta = randn_cache[address]
        else:
            eta = randn_cache[address] = torch.randn(
                y_mean.shape,
                dtype=y_mean.dtype,
                device=y_mean.device,
            )

        y = y_mean + y_var.sqrt() * eta

        return y

    def kl_divergence(self, prior_var: float = 1.0) -> Tensor:
        r"""Returns the KL divergence between the posterior and the prior."""

        kl = 0.0

        for key in self.means.keys():
            mean = self.means[key]
            log_var = _softclip(self.logvars[key], 11.0)
            var = log_var.exp()

            # fmt: off
            kl = kl + 0.5 * (
                var / prior_var
                + mean.square() / prior_var
                + math.log(prior_var) - log_var
                - 1.0
            ).sum()
            # fmt: on

        return kl
