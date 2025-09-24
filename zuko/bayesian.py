r"""Utils for bayesian flows."""

from __future__ import annotations

__all__ = [
    "BayesianModel",
]

import math
import torch
import torch.nn as nn

from contextlib import contextmanager
from torch.func import functional_call

from .nn import Linear, MaskedLinear


class BayesianModel(nn.Module):
    def __init__(self, base: nn.Module, init_logvar: float = -9.0, learn_means: bool = False):
        super().__init__()
        self.base = base
        self.learn_means = learn_means
        self.init_logvar = init_logvar

        # Store parameters for Bayesian layers - use underscores instead of dots
        self.weight_means = nn.ParameterDict()
        self.bias_means = nn.ParameterDict()
        self.weight_logvars = nn.ParameterDict()
        self.bias_logvars = nn.ParameterDict()

        for name, module in base.named_modules():
            if isinstance(module, (Linear, MaskedLinear)):
                # Convert dots to underscores for ParameterDict keys
                safe_name = name.replace(".", "_")

                # Only store mean parameters if learn_means=True
                if learn_means:
                    self.weight_means[safe_name + "_weight"] = nn.Parameter(
                        module.weight.detach().clone()
                    )
                    if module.bias is not None:
                        self.bias_means[safe_name + "_bias"] = nn.Parameter(
                            module.bias.detach().clone()
                        )

                # Always store log-variance parameters
                self.weight_logvars[safe_name + "_weight"] = nn.Parameter(
                    torch.full_like(module.weight, init_logvar)
                )
                if module.bias is not None:
                    self.bias_logvars[safe_name + "_bias"] = nn.Parameter(
                        torch.full_like(module.bias, init_logvar)
                    )

        # Initialize them properly
        self._reset_bayesian_parameters()

    def _reset_bayesian_parameters(self):
        """Initialize posterior means and variances like in original impl."""
        for name, module in self.base.named_modules():
            if isinstance(module, (Linear, MaskedLinear)):
                safe_name = name.replace(".", "_")
                fan_in = module.weight.size(-1)
                stdv = 1.0 / math.sqrt(fan_in)

                # Initialize base model weights
                if module.bias is not None:
                    module.bias.data.zero_()

                # Weight mean
                if safe_name + "_weight" in self.weight_means:
                    self.weight_means[safe_name + "_weight"].data.normal_(0, stdv)
                else:
                    module.weight.data.normal_(0, stdv)

                # Weight logvar - match BayesianLinear initialization pattern
                self.weight_logvars[safe_name + "_weight"].data.zero_().normal_(
                    self.init_logvar, 0.001
                )

                # Bias mean
                if module.bias is not None:
                    if safe_name + "_bias" in self.bias_means:
                        self.bias_means[safe_name + "_bias"].data.zero_()
                    else:
                        module.bias.data.zero_()

                # Bias logvar - match BayesianLinear initialization pattern
                if module.bias is not None:
                    self.bias_logvars[safe_name + "_bias"].data.zero_().normal_(
                        self.init_logvar, 0.001
                    )

    def _sample_params(self):
        """Return sampled parameter dict {full_name: tensor}."""
        sampled = {}
        sampled_count = 0

        for name, module in self.base.named_modules():
            if isinstance(module, (Linear, MaskedLinear)):
                safe_name = name.replace(".", "_")

                # Weights - always read from base model unless learn_means=True
                if self.learn_means and safe_name + "_weight" in self.weight_means:
                    w_mu = self.weight_means[safe_name + "_weight"]
                    # print(f"[BayesianFlow._sample_params] Using stored weight mean for '{name}' (shape={w_mu.shape})")
                else:
                    w_mu = module.weight

                # Apply same clamping as BayesianLinear and use sqrt of exp(logvar)
                w_logvar = self.weight_logvars[safe_name + "_weight"].clamp(-11, 11)
                w_std = torch.exp(w_logvar)
                sampled[name + ".weight"] = w_mu + w_std.sqrt() * torch.randn_like(w_mu)
                sampled_count += 1

                # Bias - always read from base model unless learn_means=True
                if module.bias is not None:
                    if self.learn_means and safe_name + "_bias" in self.bias_means:
                        b_mu = self.bias_means[safe_name + "_bias"]
                    else:
                        b_mu = module.bias

                    # Apply same clamping as BayesianLinear and use sqrt of exp(logvar)
                    b_logvar = self.bias_logvars[safe_name + "_bias"].clamp(-11, 11)
                    b_std = torch.exp(b_logvar)
                    sampled[name + ".bias"] = b_mu + b_std.sqrt() * torch.randn_like(b_mu)
                    sampled_count += 1

        return sampled

    @contextmanager
    def sample(self):
        """Context manager yielding a proxy model with sampled parameters."""
        # print(f"[BayesianFlow.sample] Starting sample context (training={self.training})")

        if self.training:
            # Use local reparameterization trick for training (more efficient)
            with self._reparameterization_context():
                yield self.base
        else:
            # Use parameter sampling for evaluation
            sampled_params = self._sample_params()
            proxy = self._create_sampled_proxy(sampled_params)
            try:
                yield proxy
            finally:
                del proxy

    def _create_sampled_proxy(self, sampled_params):
        """Create proxy with pre-sampled parameters."""

        class SampledProxy(nn.Module):
            def __init__(self, base, params):
                super().__init__()
                self._base = base
                self._params = params

            def __getattr__(self, name):
                # delegate attributes (log_prob, transform, etc.) to base
                if name in {"_base", "_params"}:
                    return super().__getattr__(name)
                return getattr(self._base, name)

            def forward(self, *args, **kwargs):
                return functional_call(self._base, self._params, args, kwargs)

        return SampledProxy(self.base, sampled_params)

    @contextmanager
    def _reparameterization_context(self):
        """Context manager that temporarily replaces linear layer forwards with reparameterized versions."""
        # Store original forward methods and replace them
        original_forwards = {}

        for name, module in self.base.named_modules():
            if isinstance(module, (Linear, MaskedLinear)):
                original_forwards[name] = module.forward

                # Create closure to capture current name and module
                def make_reparameterized_forward(module_name, original_module):
                    def reparameterized_forward(input):
                        return self._apply_local_reparameterization(
                            input, module_name, original_module
                        )

                    return reparameterized_forward

                # Replace the forward method using setattr to avoid type checker issues
                module.forward = make_reparameterized_forward(name, module)

        try:
            yield
        finally:
            # Restore original forward methods
            for name, module in self.base.named_modules():
                if name in original_forwards:
                    module.forward = original_forwards[name]

    def _apply_local_reparameterization(self, input, name, module):
        """Apply local reparameterization trick to a linear layer."""
        safe_name = name.replace(".", "_")

        # Get mean weights
        if self.learn_means and safe_name + "_weight" in self.weight_means:
            w_mu = self.weight_means[safe_name + "_weight"]
        else:
            w_mu = module.weight

        # Get bias mean
        if module.bias is not None:
            if self.learn_means and safe_name + "_bias" in self.bias_means:
                b_mu = self.bias_means[safe_name + "_bias"]
            else:
                b_mu = module.bias
        else:
            b_mu = None

        # Compute mean output
        if isinstance(module, MaskedLinear):
            mu_out = nn.functional.linear(input, module.mask * w_mu, b_mu)
        else:
            mu_out = nn.functional.linear(input, w_mu, b_mu)

        # Get weight variances
        w_logvar = self.weight_logvars[safe_name + "_weight"].clamp(-11, 11)
        w_var = torch.exp(w_logvar)

        # Compute variance of output using local reparameterization trick
        if isinstance(module, MaskedLinear):
            var_out = nn.functional.linear(input.pow(2), module.mask * w_var) + 1e-8
        else:
            var_out = nn.functional.linear(input.pow(2), w_var) + 1e-8

        # Add bias variance if present
        if module.bias is not None:
            b_logvar = self.bias_logvars[safe_name + "_bias"].clamp(-11, 11)
            b_var = torch.exp(b_logvar)
            if isinstance(module, MaskedLinear):
                # Apply mask to bias variance - use mask's output dimension
                masked_b_var = module.mask.any(dim=1).float() * b_var
                var_out = var_out + masked_b_var.unsqueeze(0).expand_as(var_out)
            else:
                var_out = var_out + b_var.unsqueeze(0).expand_as(var_out)

        # Sample output using reparameterization trick
        result = mu_out + var_out.sqrt() * torch.randn_like(mu_out)
        return result

    def kl_divergence(self, prior_std: float = 1.0, clamp: tuple = (-11, 11)):
        """Compute KL divergence between q(w|θ) and prior N(0, σ_p^2)."""
        kl = 0.0
        prior_var = prior_std**2
        log_prior_var = math.log(prior_var)
        layer_count = 0

        for name, module in self.base.named_modules():
            if isinstance(module, (Linear, MaskedLinear)):
                layer_count += 1
                safe_name = name.replace(".", "_")

                # Weights - always read from base model unless learn_means=True
                if self.learn_means and safe_name + "_weight" in self.weight_means:
                    mu = self.weight_means[safe_name + "_weight"]
                else:
                    mu = module.weight

                logvar = self.weight_logvars[safe_name + "_weight"].clamp(*clamp)
                post_var = torch.exp(logvar)

                weight_kl = (
                    0.5
                    * (
                        (post_var / prior_var) + (mu**2) / prior_var - 1.0 + log_prior_var - logvar
                    ).sum()
                )
                kl += weight_kl

                # Biases - always read from base model unless learn_means=True
                if module.bias is not None:
                    if self.learn_means and safe_name + "_bias" in self.bias_means:
                        mu = self.bias_means[safe_name + "_bias"]
                    else:
                        mu = module.bias

                    logvar = self.bias_logvars[safe_name + "_bias"].clamp(*clamp)
                    post_var = torch.exp(logvar)

                    bias_kl = (
                        0.5
                        * (
                            (post_var / prior_var)
                            + (mu**2) / prior_var
                            - 1.0
                            + log_prior_var
                            - logvar
                        ).sum()
                    )
                    kl += bias_kl

        return kl

    def forward(self, *args, **kwargs):
        # Default deterministic call: use base weights (means)
        result = self.base(*args, **kwargs)
        return result
