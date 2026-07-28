# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Compute Colorfulness of an image.

Colorfullness is defined by Hasler-Süsstrunk metric
"""

import cv2
import numpy as np
import torch

from infqm.base_metric import BaseMetric


class ColorfulnessMetric(BaseMetric):
    """Class to compute Colorfulness of an image."""

    def calculate(self, cv_image):
        """Calculate colorfulness using Hasler-Süsstrunk metric.

        Fast perceptual colorfulness measure

        Args:
            cv_image: OpenCV image (numpy array).

        Returns:
            float - normalized colorfulness (0-1)
        """
        if len(cv_image.shape) != 3:
            return 0.0  # Grayscale has no colorfulness

        # Convert BGR to RGB for correct color analysis
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB) / 255.0
        red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

        # Calculate rg and yb opponent colors
        rg = red.astype(np.float32) - green.astype(np.float32)
        yb = 0.5 * (red.astype(np.float32) + green.astype(np.float32)) - blue.astype(
            np.float32
        )

        # Calculate standard deviations and means
        rg_std, rg_mean = np.std(rg), np.mean(np.abs(rg))
        yb_std, yb_mean = np.std(yb), np.mean(np.abs(yb))

        # Hasler-Süsstrunk colorfulness formula
        std_root = np.sqrt(rg_std**2 + yb_std**2)
        mean_root = np.sqrt(rg_mean**2 + yb_mean**2)

        colorfulness = std_root + 0.3 * mean_root

        # Normalize to 0-1 range (typical max around 100-150 for natural images)
        return min(colorfulness, 1.0)

    def calculate_tensor(self, image_batch):
        """Calculate colorfulness using Hasler-Süsstrunk metric Fast perceptual
        colorfulness measure.

        Args:
            image_batch: PyTorch tensor of shape (C, H, W) or (H, W)

        Returns:
            Tensor of shape (batch_size,) containing colorfulness for each image.
        """
        # Assuming tensor is in RGB format with shape (batch_size, 3, H, W)
        red = image_batch[:, 0]  # Shape: (batch_size, H, W)
        green = image_batch[:, 1]
        blue = image_batch[:, 2]

        # Calculate rg and yb opponent colors
        rg = red - green  # Shape: (batch_size, H, W)
        yb = 0.5 * (red + green) - blue

        # Calculate standard deviations and means per image
        # We need to compute over spatial dimensions (H, W) for each image
        rg_std = torch.std(
            rg.view(image_batch.shape[0], -1), dim=1
        )  # Shape: (batch_size,)
        rg_mean = torch.mean(torch.abs(rg.view(image_batch.shape[0], -1)), dim=1)
        yb_std = torch.std(yb.view(image_batch.shape[0], -1), dim=1)
        yb_mean = torch.mean(torch.abs(yb.view(image_batch.shape[0], -1)), dim=1)

        # Hasler-Süsstrunk colorfulness formula
        std_root = torch.sqrt(rg_std**2 + yb_std**2)
        mean_root = torch.sqrt(rg_mean**2 + yb_mean**2)
        colorfulness_values = std_root + 0.3 * mean_root

        # Normalize and clamp
        normalized = colorfulness_values
        return torch.clamp(normalized, max=1.0)
