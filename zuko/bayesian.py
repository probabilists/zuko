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
    def __init__(
        self,
        base: nn.Module,
        init_logvar: float = -9.0,
        learn_means: bool = False,
        bayesian_layers: list[str] | None = None,
    ):
        super().__init__()
        self.base = base
        self.learn_means = learn_means
        self.init_logvar = init_logvar
        # bayesian_layers: optional list of module names (dotted, e.g. "layer1.linear")
        # If None, all Linear/MaskedLinear modules are treated as Bayesian (existing behavior).
        # Store both dotted and underscore-safe names for easy matching.
        if bayesian_layers is None:
            self._bayesian_names = None
        else:
            safe = set()
            for n in bayesian_layers:
                safe.add(n)
            self._bayesian_names = safe
        # Keep original requested names for validation/warnings
        self._requested_bayesian = set(bayesian_layers) if bayesian_layers is not None else None

        # Store parameters for Bayesian layers - use underscores instead of dots
        self.weight_means = nn.ParameterDict()
        self.bias_means = nn.ParameterDict()
        self.weight_logvars = nn.ParameterDict()
        self.bias_logvars = nn.ParameterDict()

        matched = set()
        for name, module in base.named_modules():
            if isinstance(module, (Linear, MaskedLinear)):
                # Convert dots to underscores for ParameterDict keys
                # If specific bayesian layers selected, skip others
                if self._bayesian_names is not None and name not in self._bayesian_names:
                    continue
                safe_name = name.replace(".", "_")
                # Mark as matched for validation
                if self._bayesian_names is not None:
                    matched.add(name)

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

        # Validate requested bayesian layer names and warn if any weren't found
        if self._requested_bayesian is not None:
            # Consider a requested name found if either dotted or underscore form matched
            found = set()
            for r in self._requested_bayesian:
                if r in matched:
                    found.add(r)
            missing = set(self._requested_bayesian) - found
            if missing:
                import warnings

                warnings.warn(
                    f"BayesianModel: requested bayesian_layers not found in base model: {sorted(missing)}",
                    UserWarning,
                    stacklevel=2,
                )

    def _reset_bayesian_parameters(self):
        """Initialize posterior means and variances like in original impl."""
        for name, module in self.base.named_modules():
            if isinstance(module, (Linear, MaskedLinear)):
                # Skip non-selected layers when a subset was provided
                if self._bayesian_names is not None and name not in self._bayesian_names:
                    continue

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
                # Skip non-selected layers when a subset was provided

                if self._bayesian_names is not None and name not in self._bayesian_names:
                    continue
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
                # Skip non-selected layers when a subset was provided
                if self._bayesian_names is not None and name not in self._bayesian_names:
                    continue

                original_forwards[name] = module.forward

                # Create closure to capture current name and module
                def make_reparameterized_forward(module_name, original_module):
                    # Match original forward signature (x) to avoid type errors
                    def reparameterized_forward(x):
                        return self._apply_local_reparameterization(
                            x, module_name, original_module
                        )

                    return reparameterized_forward

                # Replace the forward method using setattr on instance to avoid mypy complaints
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
        # Skip non-selected layers when a subset was provided (shouldn't happen if context applied correctly)
        if self._bayesian_names is not None and name not in self._bayesian_names:
            # Fallback to default linear forward
            return module.forward(input)

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
                # Skip non-selected layers when a subset was provided
                if self._bayesian_names is not None and name not in self._bayesian_names:
                    continue

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
