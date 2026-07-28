# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Description."""

import cv2
import numpy as np
import torch

from infqm.base_metric import BaseMetric


class SignalNoiseMetric(BaseMetric):
    """Class to compute Signal-to-Noise Ratio of an image."""

    def calculate(self, cv_image):
        """Calculate Signal-to-Noise Ratio. Uses mean/std as a simple SNR
        approximation.

        Args:
            cv_image: OpenCV image (numpy array).

        Returns:
            float - SNR value (higher = better quality)
        """
        # Convert to grayscale if color image
        if len(cv_image.shape) == 3:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv_image

        mean_signal = np.mean(gray)
        std_noise = np.std(gray)

        # Avoid division by zero
        if std_noise == 0:
            return float("inf")

        snr = mean_signal / (std_noise + 1e-8)
        return snr / (snr + 1)

    def calculate_tensor(self, image_batch):
        """Calculate Signal-to-Noise Ratio Uses mean/std as a simple SNR
        approximation.

        Args:
            image_batch: PyTorch tensor of shape (C, H, W) or (H, W).

        Returns:
            tensor - SNR value (higher = better quality).
        """
        # Convert to grayscale if color image
        gray = self.tensor_rgb2gray(image_batch=image_batch)

        gray_flat = gray.view(gray.shape[0], -1)

        mean_signal = torch.mean(gray_flat, dim=1)  # Shape: (batch_size,)
        std_noise = torch.std(gray_flat, dim=1)  # Shape: (batch_size,)

        # Avoid division by zero
        snr = torch.where(
            std_noise == 0,
            torch.tensor(float("inf"), dtype=gray.dtype, device=gray.device),
            mean_signal / std_noise,
        )

        # Normalize SNR to [0, 1] range
        return snr / (snr + 1)
