# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Description."""

import cv2
import numpy as np
import torch

from infqm.base_metric import BaseMetric


class SaturationMetric(BaseMetric):
    """Class to compute saturation of an image."""

    def calculate(self, cv_image):
        """Calculate saturation of an image.

        Args:
            cv_image: OpenCV image (numpy array).

        Returns:
            float - saturation value (0-1, higher = more saturated).
        """
        if len(cv_image.shape) == 2:  # grayscale image has no saturation
            return 0.0
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1].astype(np.float32) / 255.0
        return float(np.mean(s))

    def calculate_tensor(self, image_batch):
        """Calculate saturation of an image tensor.

        Args:
            image_batch: PyTorch tensor of shape (batch_size, 3, H, W).

        Returns:
            tensor - saturation values (batch_size,) in range [0, 1].
        """
        # Assuming RGB format with shape (batch_size, 3, H, W)
        red = image_batch[:, 0]  # Shape: (batch_size, H, W)
        green = image_batch[:, 1]
        blue = image_batch[:, 2]

        # Convert RGB to HSV - extract saturation channel
        # Stack along a new dimension for min/max operations
        rgb_stack = torch.stack(
            [red, green, blue], dim=1
        )  # Shape: (batch_size, 3, H, W)
        max_rgb = torch.max(rgb_stack, dim=1)[0]  # Shape: (batch_size, H, W)
        min_rgb = torch.min(rgb_stack, dim=1)[0]  # Shape: (batch_size, H, W)

        # Saturation = (max - min) / max, avoid division by zero
        delta = max_rgb - min_rgb
        s = torch.where(max_rgb > 0, delta / max_rgb, torch.zeros_like(max_rgb))
        # s has shape: (batch_size, H, W)

        # Calculate mean saturation per image (average over H and W dimensions)
        return torch.mean(s, dim=(1, 2))  # Shape: (batch_size,)
