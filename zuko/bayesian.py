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

from .flows import GF, NAF
from .nn import Linear, MaskedLinear, MonotonicLinear, linear


def _softclip(x, bound=11.0):
    """Soft clipping function to keep x in [-bound, bound]."""
    return x * (1 + (x / bound).square()).rsqrt()


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

        # bayesian_layers: optional list of module names (dotted, e.g. "layer1.linear")
        # If None, all Linear/MaskedLinear modules are treated as Bayesian (existing behavior).
        if bayesian_layers is None:
            self._requested_bayesian = None
        else:
            self._requested_bayesian = set(bayesian_layers)

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
                if self._requested_bayesian is not None and name not in self._requested_bayesian:
                    continue
                safe_name = name.replace(".", "_")
                # Mark as matched for validation
                if self._requested_bayesian is not None:
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
        self._reset_bayesian_parameters(init_logvar)

        # Validate requested bayesian layer names and warn if any weren't found
        if self._requested_bayesian is not None:
            missing = self._requested_bayesian - matched
            if missing:
                warnings.warn(
                    f"BayesianModel: requested bayesian_layers not found in base model: {sorted(missing)}",
                    UserWarning,
                    stacklevel=2,
                )

    def _reset_bayesian_parameters(self, init_logvar: float):
        """Initialize posterior means and variances like in original impl."""
        for name, module in self.base.named_modules():
            if isinstance(module, (Linear, MaskedLinear)):
                # Skip non-selected layers when a subset was provided
                if self._requested_bayesian is not None and name not in self._requested_bayesian:
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
                self.weight_logvars[safe_name + "_weight"].data.zero_().normal_(init_logvar, 0.001)

                # Bias mean
                if module.bias is not None:
                    if safe_name + "_bias" in self.bias_means:
                        self.bias_means[safe_name + "_bias"].data.zero_()
                    else:
                        module.bias.data.zero_()

                # Bias logvar - match BayesianLinear initialization pattern
                if module.bias is not None:
                    self.bias_logvars[safe_name + "_bias"].data.zero_().normal_(init_logvar, 0.001)

    def _sample_params(self):
        """Return sampled parameter dict {full_name: tensor}."""
        sampled = {}

        for name, module in self.base.named_modules():
            if isinstance(module, (Linear, MaskedLinear)):
                # Skip non-selected layers when a subset was provided
                if self._requested_bayesian is not None and name not in self._requested_bayesian:
                    continue
                safe_name = name.replace(".", "_")

                # Weights - always read from base model unless learn_means=True
                if self.learn_means and safe_name + "_weight" in self.weight_means:
                    w_mu = self.weight_means[safe_name + "_weight"]
                else:
                    w_mu = module.weight

                # Apply same softclip as BayesianLinear and use sqrt of exp(logvar)
                w_logvar = _softclip(self.weight_logvars[safe_name + "_weight"], 11)
                sampled[name + ".weight"] = w_mu + torch.exp(w_logvar).sqrt() * torch.randn_like(
                    w_mu
                )

                # Bias - always read from base model unless learn_means=True
                if module.bias is not None:
                    if self.learn_means and safe_name + "_bias" in self.bias_means:
                        b_mu = self.bias_means[safe_name + "_bias"]
                    else:
                        b_mu = module.bias

                    # Apply same softclip as BayesianLinear and use sqrt of exp(logvar)
                    b_logvar = _softclip(self.bias_logvars[safe_name + "_bias"], 11)
                    sampled[name + ".bias"] = b_mu + torch.exp(b_logvar).sqrt() * torch.randn_like(
                        b_mu
                    )

        return sampled

    @contextmanager
    def sample(self, trick=False):
        """Context manager yielding the model with reparametrized forward methods for linear model.
        This is not cloning the model but just temporarily replacing the forward methods of the
        selected linear layers to use the local reparameterization trick during training, or
        sampling once during evaluation.

        Arguments:
            trick: if True, use the local reparameterization trick even in eval mode."""

        if (not trick and not self.training) or isinstance(self.base, (NAF, GF)):
            sampled_params = self._sample_params()
            with torch.nn.utils.stateless._reparametrize_module(self.base, sampled_params):
                yield self.base
        else:
            with self._reparameterization_context():
                yield self.base

    @contextmanager
    def _reparameterization_context(self):
        """Context manager that temporarily replaces linear layer forwards with reparameterized versions."""
        # Store original forward methods and replace them
        original_forwards = {}

        # Sample a new seed for each module call
        for name, module in self.base.named_modules():
            if isinstance(module, (Linear, MaskedLinear)):
                # Skip non-selected layers when a subset was provided
                if self._requested_bayesian is not None and name not in self._requested_bayesian:
                    continue

                seed = torch.randint(0, 2**31 - 1, ()).item()
                original_forwards[name] = module.forward

                module.forward = partial(
                    self._apply_local_reparameterization_training,
                    name=name,
                    module=module,
                    seed=seed,
                )
        try:
            yield
        finally:
            # Restore original forward methods
            for name, module in self.base.named_modules():
                if name in original_forwards:
                    module.forward = original_forwards[name]

    def _apply_local_reparameterization_training(self, input, name, module, seed):
        """Apply local reparameterization trick to a linear layer."""
        # Skip non-selected layers when a subset was provided (shouldn't happen if context applied correctly)
        if self._requested_bayesian is not None and name not in self._requested_bayesian:
            # Fallback to default linear forward
            return module.forward(input)

        safe_name = name.replace(".", "_")

        # each model has its generator, which get reset to the same seed at each call of forward
        generator = torch.Generator(device=input.device).manual_seed(seed)

        # Get mean weights
        if self.learn_means and safe_name + "_weight" in self.weight_means:
            w_mu = self.weight_means[safe_name + "_weight"]
        else:
            w_mu = module.weight

        if isinstance(module, MonotonicLinear):
            w_mu = w_mu.abs()

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
            mu_out = linear(input, module.mask * w_mu, b_mu)
        else:
            mu_out = linear(input, w_mu, b_mu)

        # Get weight variances
        w_logvar = _softclip(self.weight_logvars[safe_name + "_weight"], 11)
        w_var = torch.exp(w_logvar)

        # Compute variance of output using local reparameterization trick
        if isinstance(module, MaskedLinear):
            var_out = linear(input.square(), module.mask * w_var) + 1e-8
        else:
            var_out = linear(input.square(), w_var) + 1e-8

        # Add bias variance if present
        if module.bias is not None:
            b_logvar = _softclip(self.bias_logvars[safe_name + "_bias"], 11)
            var_out = var_out + torch.exp(b_logvar)

        # Sample output using reparameterization trick
        result = torch.normal(mu_out, var_out.sqrt(), generator=generator)
        return result

    def kl_divergence(self, prior_std: float = 1.0):
        """Compute KL divergence between q(w|θ) and prior N(0, σ_p^2)."""
        kl = 0.0
        prior_var = prior_std**2
        log_prior_var = math.log(prior_var)

        for name, module in self.base.named_modules():
            if isinstance(module, (Linear, MaskedLinear)):
                # Skip non-selected layers when a subset was provided
                if self._requested_bayesian is not None and name not in self._requested_bayesian:
                    continue

                safe_name = name.replace(".", "_")

                # Weights - always read from base model unless learn_means=True
                if self.learn_means and safe_name + "_weight" in self.weight_means:
                    mu = self.weight_means[safe_name + "_weight"]
                else:
                    mu = module.weight

                logvar = _softclip(self.weight_logvars[safe_name + "_weight"], 11.0)
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

                    logvar = _softclip(self.bias_logvars[safe_name + "_bias"], 11.0)
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

    def sample_model(self):
        """Return a single sampled model instance (not a context manager).
        This effectively clones the base model and samples new parameters for the
        selected Bayesian layers.

        Returns:
            A clone of  model with sampled parameters that can be used like the original model.
        """
        sampled_params = self._sample_params()
        # Clone the base model and apply sampled parameters
        # deep copy of the base model
        new_model = copy.deepcopy(self.base)
        new_model.load_state_dict(sampled_params, strict=False)
        return new_model

    def forward(self, *args, **kwargs):
        """Exception: the model should not be used directly"""
        raise RuntimeError(
            "BayesianModel should not be used directly. Use sample() context manager or sample_model()."
        )
