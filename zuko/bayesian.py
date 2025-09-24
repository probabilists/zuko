r"""Utils for bayesian flows."""

from __future__ import annotations

__all__ = [
    "BayesianModel",
]
import copy
import math
import torch
import torch.nn as nn
import warnings

from contextlib import contextmanager
from functools import partial
from torch import Tensor
from typing import Dict, Generator, Sequence

from .nn import Linear, MaskedLinear, MonotonicLinear, linear


def _softclip(x: Tensor, bound: float) -> Tensor:
    return x * (1 + (x / bound).square()).rsqrt()


class BayesianModel(nn.Module):
    r"""Creates a Bayesian wrapper around a base model.

    Currently, only linear layers are supported.

    Arguments:
        base: A base model.
        bayesian_layers: A subset of layer names to target. The names should match those
            returned by `base.named_modules()`.
    """

    def __init__(
        self,
        base: nn.Module,
        bayesian_layers: Sequence[str] = (),
    ):
        super().__init__()

        self.base = base
        self.bayesian_layers = set(bayesian_layers)

        self.means = nn.ParameterDict()
        self.logvars = nn.ParameterDict()

        layers = set()

        for name, module in self.base.named_modules():
            if self.bayesian_layers and name not in self.bayesian_layers:
                continue
            elif not isinstance(module, (nn.Linear, Linear, MaskedLinear)):
                continue
            elif isinstance(module, MonotonicLinear):
                continue

            layers.add(name)

            key = name.replace(".", "-")

            self.means[key + "-weight"] = nn.Parameter(torch.empty_like(module.weight))
            self.logvars[key + "-weight"] = nn.Parameter(torch.empty_like(module.weight))

            if module.bias is not None:
                self.means[key + "-bias"] = nn.Parameter(torch.empty_like(module.bias))
                self.logvars[key + "-bias"] = nn.Parameter(torch.empty_like(module.bias))

        if self.bayesian_layers:
            missing = sorted(self.bayesian_layers - layers)
            if missing:
                warnings.warn(
                    f"BayesianModel: some layers were not found in the base model:\n\t{missing}",
                    category=UserWarning,
                    stacklevel=2,
                )
        else:
            self.bayesian_layers = layers

        self._reset_bayesian_parameters()

    def _reset_bayesian_parameters(self, logvar_mean: float = -9.0, logvar_std: float = 1e-3):
        r"""Initializes the posterior means and log-variances."""

        for name, module in self.base.named_modules():
            if name not in self.bayesian_layers:
                continue

            key = name.replace(".", "-")

            self.means[key + "-weight"].data.copy_(module.weight.data)
            self.logvars[key + "-weight"].data.normal_(logvar_mean, logvar_std)

            if module.bias is not None:
                self.means[key + "-bias"].data.copy_(module.bias.data)
                self.logvars[key + "-bias"].data.normal_(logvar_mean, logvar_std)

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "BayesianModel should not be called directly. Use `sample_model` or `reparameterize` instead."
        )

    def sample_params(self) -> Dict[str, Tensor]:
        r"""Returns model parameters sampled from the posterior."""

        params = {}

        for key in self.means.keys():
            mean = self.means[key]
            std = _softclip(self.logvars[key], 11.0).div(2.0).exp()

            name = key.replace("-", ".")
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

        Within this context, the base model is temporarily reparametrized and
        behaves deterministically.

        Arguments:
            local_trick: Whether to use the local reparameterization trick.

        Yields:
            The reparametrized base model.

        References:
            | Variational Dropout and the Local Reparameterization Trick (Kingma et al., 2015)
            | https://arxiv.org/abs/1506.02557
        """

        if local_trick:
            with self._reparametrize_trick():
                yield self.base
        else:
            params = self.sample_params()
            with torch.nn.utils.stateless._reparametrize_module(self.base, params):
                yield self.base

    @contextmanager
    def _reparametrize_trick(self):
        original_forwards = {}

        for name, module in self.base.named_modules():
            if name in self.bayesian_layers:
                seed = torch.randint(0, 2**31 - 1, (), device="cpu").item()
                original_forwards[name] = module.forward
                module.forward = partial(
                    self._local_reparameterization_trick,
                    name=name,
                    module=module,
                    seed=seed,
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
        seed: int,
    ) -> Tensor:
        """Applies the local reparameterization trick to a linear layer."""

        generator = torch.Generator(device=x.device).manual_seed(seed)

        key = name.replace(".", "-")

        w_mean = self.means[key + "-weight"]
        w_var = _softclip(self.logvars[key + "-weight"], 11.0).exp()

        if isinstance(module, MaskedLinear):
            w_mean = module.mask * w_mean
            w_var = module.mask * w_var

        if module.bias is not None:
            b_mean = self.means[key + "-bias"]
            b_var = _softclip(self.logvars[key + "-bias"], 11.0).exp()
        else:
            b_mean = None
            b_var = None

        y_mean = linear(x, w_mean, b_mean)
        y_var = linear(x.square(), w_var, b_var)

        y = y_mean + y_var.sqrt() * torch.randn(
            y_mean.shape,
            dtype=y_mean.dtype,
            device=y_mean.device,
            generator=generator,
        )

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
