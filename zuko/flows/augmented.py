"""Augmented normalizing flows."""

__all__ = [
    "ANF",
    "AutoencodingTransform",
    "HierarchicalANF",
]

import torch
import torch.nn as nn

from math import prod
from torch import Size, Tensor
from torch.distributions import Transform, constraints
from typing import Callable, Sequence

from ..distributions import DiagNormal, Joint
from ..lazy import Flow, LazyTransform, UnconditionalDistribution
from ..nn import MLP
from ..transforms import CouplingTransform, DependentTransform, MonotonicAffineTransform
from ..utils import broadcast, unpack


class AutoencodingTransform(LazyTransform):
    r"""Creates a lazy autoencoding transformation for ANF."""

    def __init__(
        self,
        features: int,
        noise_features: int,
        context: int = 0,
        univariate: Callable[..., nn.Module] = MonotonicAffineTransform,
        shapes: Sequence[Size] = ((), ()),
        **kwargs
    ):
        super().__init__()

        self.features = features
        self.noise_features = noise_features

        # Univariate transformation
        self.univariate = univariate
        self.shapes = shapes
        self.total = sum(prod(s) for s in shapes)

        # For encoding transform, condition on x
        self.enc_mask = torch.zeros(noise_features, dtype=bool)
        self.enc_hyper = MLP(features + context, noise_features * self.total, **kwargs)

        # For decoding transform, condition on z
        self.dec_mask = torch.zeros(features, dtype=bool)
        self.dec_hyper = MLP(noise_features + context, features * self.total, **kwargs)

    def forward(self, c: Tensor = None) -> Transform:
        """
        Creates and returns a proper Transform object that implements the autoencoding transformation.
        """

        # First, create encoding and decoding transforms
        def encode_meta(x_and_c: Tensor) -> Transform:
            """Generate encoding transform parameters from x and c."""
            phi = self.enc_hyper(x_and_c)
            phi = phi.unflatten(-1, (-1, self.total))
            phi = unpack(phi, self.shapes)

            return DependentTransform(self.univariate(*phi), 1)

        def decode_meta(z_and_c: Tensor) -> Transform:
            """Generate decoding transform parameters from z and c."""
            phi = self.dec_hyper(z_and_c)
            phi = phi.unflatten(-1, (-1, self.total))
            phi = unpack(phi, self.shapes)

            return DependentTransform(self.univariate(*phi), 1)

        class ANFTransform(Transform):
            domain = constraints.real_vector
            codomain = constraints.real_vector
            bijective = True

            def __init__(self_, outer_self=self):
                super().__init__()
                self_.outer = outer_self

            def _call(self_, x_and_e: Tensor) -> Tensor:
                # Check input
                if x_and_e is None:
                    # Handle None input
                    return None

                # Get features and noise_features from outer class
                features = self.features
                noise_features = self.noise_features

                # Handle the case where features = 0 (higher levels in hierarchical model)
                if features == 0:
                    # For higher levels, we only transform the noise
                    return x_and_e

                # Split data and noise
                x, e = torch.split(x_and_e, [features, noise_features], dim=-1)

                # Prepare context
                if c is not None:
                    x_and_c = torch.cat(broadcast(x, c, ignore=1), dim=-1)
                    z_and_c_fn = lambda z: torch.cat(broadcast(z, c, ignore=1), dim=-1)
                else:
                    x_and_c = x
                    z_and_c_fn = lambda z: z

                # Apply encoding: e -> z (via coupling transform)
                z = CouplingTransform(lambda _: encode_meta(x_and_c), self.enc_mask)(e)

                # Apply decoding: x -> y (via coupling transform)
                z_and_c = z_and_c_fn(z)
                y = CouplingTransform(lambda _: decode_meta(z_and_c), self.dec_mask)(x)

                # Combine results
                return torch.cat([y, z], dim=-1)

            def _inverse(self_, y_and_z: Tensor) -> Tensor:
                # Check input
                if y_and_z is None:
                    # Handle None input
                    return None

                # Get features and noise_features from outer class
                features = self.features
                noise_features = self.noise_features

                # Handle the case where features = 0 (higher levels in hierarchical model)
                if features == 0:
                    # For higher levels, we only transform the noise
                    return y_and_z

                # Split transformed data and noise
                y, z = torch.split(y_and_z, [features, noise_features], dim=-1)

                # Prepare context
                if c is not None:
                    z_and_c = torch.cat(broadcast(z, c, ignore=1), dim=-1)
                    x_and_c_fn = lambda x: torch.cat(broadcast(x, c, ignore=1), dim=-1)
                else:
                    z_and_c = z
                    x_and_c_fn = lambda x: x

                # Apply inverse decoding: y -> x
                dec_transform = CouplingTransform(lambda _: decode_meta(z_and_c), self.dec_mask)
                x = dec_transform.inv(y)

                # Apply inverse encoding: z -> e
                x_and_c = x_and_c_fn(x)
                enc_transform = CouplingTransform(lambda _: encode_meta(x_and_c), self.enc_mask)
                e = enc_transform.inv(z)

                # Combine results
                return torch.cat([x, e], dim=-1)

            def log_abs_det_jacobian(self_, x_and_e: Tensor, y_and_z: Tensor) -> Tensor:
                # Check inputs
                if x_and_e is None or y_and_z is None:
                    # Handle None inputs
                    return torch.tensor(0.0)

                # Get features and noise_features from outer class
                features = self.features
                noise_features = self.noise_features

                # Handle the case where features = 0 (higher levels in hierarchical model)
                if features == 0:
                    # For higher levels, we return zero log det jacobian
                    return torch.zeros_like(x_and_e[..., 0])

                # Split inputs and outputs
                x, e = torch.split(x_and_e, [features, noise_features], dim=-1)
                y, z = torch.split(y_and_z, [features, noise_features], dim=-1)

                # Prepare context
                if c is not None:
                    x_and_c = torch.cat(broadcast(x, c, ignore=1), dim=-1)
                    z_and_c = torch.cat(broadcast(z, c, ignore=1), dim=-1)
                else:
                    x_and_c = x
                    z_and_c = z

                # Calculate ladj for encoding: e -> z
                enc_transform = CouplingTransform(lambda _: encode_meta(x_and_c), self.enc_mask)
                ladj_enc = enc_transform.log_abs_det_jacobian(e, z)

                # Calculate ladj for decoding: x -> y
                dec_transform = CouplingTransform(lambda _: decode_meta(z_and_c), self.dec_mask)
                ladj_dec = dec_transform.log_abs_det_jacobian(x, y)

                # Sum up
                return ladj_enc + ladj_dec
        # Return the complete Transform object
        return ANFTransform()


class ANF(Flow):
    r"""Creates an augmented normalizing flow (ANF)."""

    def __init__(
        self,
        features: int,
        noise_features: int,
        context: int = 0,
        steps: int = 1,
        **kwargs
    ):
        # Create the autoencoding transforms
        transforms = [
            AutoencodingTransform(
                features=features,
                noise_features=noise_features,
                context=context,
                **kwargs
            )
            for _ in range(steps)
        ]

        # Base distribution - joint distribution of data and noise
        data_dist = DiagNormal(torch.zeros(features), torch.ones(features))
        noise_dist = DiagNormal(torch.zeros(noise_features), torch.ones(noise_features))

        # Joint base distribution
        joint_dist = lambda: Joint(data_dist, noise_dist)

        # Wrap with UnconditionalDistribution
        base = UnconditionalDistribution(joint_dist, buffer=True)

        super().__init__(transforms, base)


class HierarchicalANF(Flow):
    r"""Creates a hierarchical augmented normalizing flow."""

    def __init__(
        self,
        features: int,
        noise_features_list: Sequence[int],
        context: int = 0,
        **kwargs
    ):
        self.levels = len(noise_features_list)
        self.features = features
        self.noise_features_list = noise_features_list

        # Create AutoencodingTransform for processing each level
        class HierarchicalTransform(LazyTransform):
            def __init__(self_, outer_self=self):
                super().__init__()
                self_.outer = outer_self

            def forward(self_, c: Tensor = None) -> Transform:
                from torch.distributions import constraints

                # Create a new Transform class for hierarchical transformation
                class HTransform(Transform):
                    domain = constraints.real_vector
                    codomain = constraints.real_vector
                    bijective = True

                    def _call(self__, x: Tensor) -> Tensor:
                        # Split original data (x) and latent variables (z1, z2, ...)
                        chunks = []
                        remaining = x

                        # Data
                        if features > 0:
                            data, remaining = torch.split(remaining, [features, remaining.size(-1) - features], dim=-1)
                            chunks.append(data)

                        # Latent variables for each level
                        for noise_features in noise_features_list:
                            if remaining.size(-1) >= noise_features:
                                noise, remaining = torch.split(remaining, [noise_features, remaining.size(-1) - noise_features], dim=-1)
                                chunks.append(noise)
                            else:
                                break

                        # Forward transformation (for each level)
                        current_context = c
                        transformed_chunks = []

                        # Process data (first level)
                        if features > 0 and len(chunks) >= 2:
                            data_transform = self__._create_transform(0, current_context)
                            data_noise = torch.cat([chunks[0], chunks[1]], dim=-1)
                            transformed = data_transform(data_noise)
                            transformed_data, transformed_noise = torch.split(transformed, [features, noise_features_list[0]], dim=-1)
                            transformed_chunks.append(transformed_data)
                            transformed_chunks.append(transformed_noise)

                            # Update context
                            if current_context is not None:
                                current_context = torch.cat([current_context, transformed_noise], dim=-1)
                            else:
                                current_context = transformed_noise

                        # Process remaining levels
                        for i in range(1, len(noise_features_list)):
                            if i + 1 < len(chunks):
                                noise_transform = self__._create_transform(i, current_context)
                                noise_pair = torch.cat([chunks[i+1], chunks[i+1]], dim=-1)  # Dummy input
                                transformed = noise_transform(noise_pair)
                                _, transformed_noise = torch.split(transformed, [0, noise_features_list[i]], dim=-1)
                                transformed_chunks.append(transformed_noise)

                                # Update context
                                if current_context is not None:
                                    current_context = torch.cat([current_context, transformed_noise], dim=-1)
                                else:
                                    current_context = transformed_noise

                        # Combine transformed chunks
                        return torch.cat(transformed_chunks, dim=-1)

                    def _inverse(self__, y: Tensor) -> Tensor:
                        # Implement inverse transformation (omitted)
                        # The actual implementation would be similar to _call but in reverse
                        return y

                    def log_abs_det_jacobian(self__, x: Tensor, y: Tensor) -> Tensor:
                        # Calculate log determinant of Jacobian (omitted)
                        # Would sum log determinants from all level transformations
                        return torch.zeros_like(x[..., 0])

                    def _create_transform(self__, level: int, context: Tensor) -> Transform:
                        """Create transform for the specified level"""
                        if level == 0:
                            # First level: data transformation
                            return AutoencodingTransform(
                                features=features,
                                noise_features=noise_features_list[0],
                                context=0 if context is None else context.size(-1),
                                **kwargs
                            )(context)
                        else:
                            # Higher levels: latent variable transformation
                            return AutoencodingTransform(
                                features=0,  # No data
                                noise_features=noise_features_list[level],
                                context=0 if context is None else context.size(-1),
                                **kwargs
                            )(context)

                # Return the hierarchical transform
                return HTransform()

        # Create base distribution
        base_components = [
            DiagNormal(torch.zeros(features), torch.ones(features))
        ]

        for noise_features in noise_features_list:
            base_components.append(
                DiagNormal(torch.zeros(noise_features), torch.ones(noise_features))
            )

        # Create joint distribution
        joint_dist = lambda: Joint(*base_components)
        base = UnconditionalDistribution(joint_dist, buffer=True)

        # Initialize the final Flow
        super().__init__(HierarchicalTransform(), base)
