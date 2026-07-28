# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Description."""

import cv2
import numpy as np
import torch
from scipy.stats import skew

from infqm.base_metric import BaseMetric


class SkewnessMetric(BaseMetric):
    """Class to compute skewness of an image."""

    def calculate(self, cv_image):
        """Calculate skewness of pixel intensity distribution.

        Args:
            cv_image: OpenCV image (numpy array).

        Returns:
            float - skewness value.
        """
        gray = (
            cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            if len(cv_image.shape) == 3
            else cv_image
        )
        flat = gray.ravel().astype(np.float32)
        return float(skew(flat))

    def calculate_tensor(self, image_batch):
        """Calculate skewness of pixel intensity distribution.

        Args:
            image_batch: PyTorch tensor of shape (batch_size, C, H, W).

        Returns:
            tensor - skewness values (batch_size,).
        """
        gray = self.tensor_rgb2gray(image_batch=image_batch)

        # Flatten spatial dimensions per image
        flat = gray.view(gray.shape[0], -1)  # Shape: (batch_size, H*W)

        # Calculate skewness per image: E[(X - μ)^3] / σ^3
        mean = torch.mean(flat, dim=1, keepdim=True)  # Shape: (batch_size, 1)
        std = torch.std(flat, dim=1, keepdim=True)  # Shape: (batch_size, 1)

        # Standardize: (X - μ) / σ
        standardized = (flat - mean) / (
            std + 1e-8
        )  # Add epsilon to avoid division by zero

        # Calculate skewness: E[(standardized)^3]
        skewness = torch.mean(standardized**3, dim=1)  # Shape: (batch_size,)

        # Handle cases where std is zero (constant images)
        return torch.where(std.squeeze(1) == 0, torch.zeros_like(skewness), skewness)
