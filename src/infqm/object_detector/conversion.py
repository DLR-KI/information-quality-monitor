# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Class for converting likelihood values to confidence values."""

import numpy as np


class Conversion:
    """Base class for converting likelihood values to confidence values."""

    def __init__(
        self, most_confident: float = 0.9, least_confident: float = 0.1
    ) -> None:
        """Save upper and lower bounds for confidence values.

        Args:
            most_confident: Value to return for very high likelihoods (default: 0.9)
            least_confident: Value to return for very low likelihoods (default: 0.1)
        """
        self.most_confident = most_confident
        self.least_confident = least_confident

    def __str__(self) -> str:
        return f"Conversion(most_confident={self.most_confident}, \
        least_confident={self.least_confident})"

    def _forward(self, x: float) -> float:
        """Forward method of the Conversion class.

        Sub-Class specific call, conversion from likelihood to
        confidence values.

        Args:
            x: Likelihood value to convert
        """
        return x

    def __repr__(self) -> str:
        return f"Conversion(most_confident={self.most_confident}, \
        least_confident={self.least_confident})"

    def __call__(self, x: float) -> float:
        """Clipping Confidence values at boundaries.

        Args:
            x: Likelihood value to convert.

        Returns:
            float: Confidence value
        """
        return max(
            self.least_confident,
            min(self.most_confident, self._forward(x)),
        )


class LinearConversion(Conversion):
    """Linear conversion from likelihood to confidence values."""

    def __init__(self, lower_likelihood_bound, upper_likelihood_bound) -> None:
        """Initialize linear conversion with specified bounds.

        Args:
            lower_likelihood_bound: Likelihood value corresponding to least confidence
            upper_likelihood_bound: Likelihood value corresponding to most confidence.
        """
        super().__init__()
        self.lower_likelihood_bound = lower_likelihood_bound
        self.upper_likelihood_bound = upper_likelihood_bound

        self.scale = (self.most_confident - self.least_confident) / (
            self.upper_likelihood_bound - self.lower_likelihood_bound
        )
        self.offset = self.least_confident - self.scale * self.lower_likelihood_bound

    def __str__(self) -> str:
        """String representation of the linear conversion.

        Returns:
            String describing the linear conversion with its parameters.
        """
        return (
            super().__str__() + f" (scale={self.scale:.3f}, offset={self.offset:.3f})"
        )

    def __repr__(self) -> str:
        """Representation of the linear conversion for debugging purposes.

        Returns:
            String describing the linear conversion with its parameters.
        """
        return (
            super().__repr__() + f" (scale={self.scale:.3f}, offset={self.offset:.3f})"
        )

    def _forward(self, x):
        """Convert likelihood to confidence using a linear function."""
        return x * self.scale + self.offset


class ArcTanConversion(Conversion):
    """ArcTan conversion from likelihood to confidence values."""

    def __init__(self, most_confident=0.7, least_confident=0.1) -> None:
        """Initialize ArcTan conversion with specified bounds.

        Args:
            most_confident: Value to return for very high likelihoods
            least_confident: Value to return for very low likelihoods.
        """
        super().__init__(most_confident, least_confident)

    def __str__(self) -> str:
        """String representation of the ArcTan conversion.

        Returns:
            String describing the ArcTan conversion with its parameters.
        """
        return super().__str__() + " (ArcTan Conversion)"

    def get_most_confident(self) -> float:
        """Get the most confident value.

        Returns:
            self.most_confident: The most confident value.
        """
        return self.most_confident

    def get_least_confident(self) -> float:
        """Get the least confident value.

        Returns:
            self.least_confident: The least confident value.
        """
        return self.least_confident

    def _forward(self, x) -> float:
        """Convert likelihood to confidence using an arctan function.

        Args:
            x: Likelihood value to convert

        Returns:
            Confidence value corresponding to the input likelihood.
        """
        return np.arctan(0.2 * (x - 20.0)) / np.pi + 0.5
