# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Description."""

import cv2
import numpy as np
import torch

from infqm.base_metric import BaseMetric


class PsnrMetric(BaseMetric):
    """Compute Peak Signal-to-Noise Ratio (PSNR) of an image."""

    def calculate(self, cv_image):
        """Calculate Peak Signal-to-Noise Ratio.

        Uses deviation from mean as a simple quality measure without reference

        Args:
            cv_image: OpenCV image (numpy array).

        Returns:
            float - normalized PSNR value (0-1, higher = better quality)
        """
        # Convert to grayscale if color image
        if len(cv_image.shape) == 3:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv_image

        # Normalize to [0, 1] if needed
        if gray.max() > 1.0:
            gray = gray.astype(np.float32) / 255.0

        max_pixel_value = 1.0

        # Calculate MSE using deviation from mean
        image_mean = np.mean(gray)
        mse = np.mean((gray - image_mean) ** 2)

        # Avoid log(0)
        epsilon = 1e-10
        mse = max(mse, epsilon)

        # PSNR formula: 10 * log10(MAX^2 / MSE)
        psnr_db = 10 * np.log10((max_pixel_value**2) / mse)

        # Normalize from typical range [20, 50] dB to [0, 1]
        min_psnr = 0.0
        max_psnr = 50.0
        normalized_psnr = (psnr_db - min_psnr) / (max_psnr - min_psnr)
        return np.clip(normalized_psnr, 0.0, 1.0)

    def calculate_tensor(self, image_batch):
        """Calculate Peak Signal-to-Noise Ratio for a batch of images Uses
        deviation from mean as a simple quality measure without reference.

        Args:
            image_batch: PyTorch tensor of shape (batch_size, C, H, W)

        Returns:
            tensor - normalized PSNR values (batch_size,) in range [0, 1].
        """
        # Convert to grayscale if needed
        gray = self.tensor_rgb2gray(image_batch=image_batch)

        # Normalize to [0, 1] if needed
        if gray.max() > 1.0:
            gray /= 255.0

        max_pixel_value = 1.0
        batch_size = gray.shape[0]

        # Flatten spatial dimensions
        image_flat = gray.view(batch_size, -1)  # Shape: (batch_size, H*W)

        # Calculate MSE using deviation from mean per image
        image_mean = torch.mean(
            image_flat, dim=1, keepdim=True
        )  # Shape: (batch_size, 1)
        mse_per_image = torch.mean(
            (image_flat - image_mean) ** 2, dim=1
        )  # Shape: (batch_size,)

        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        mse_per_image = torch.clamp(mse_per_image, min=epsilon)

        # PSNR formula: 10 * log10(MAX^2 / MSE)
        psnr_db = 10 * torch.log10((max_pixel_value**2) / mse_per_image)

        # Normalize from typical range [20, 50] dB to [0, 1]
        min_psnr = 0.0
        max_psnr = 50.0
        normalized_psnr = (psnr_db - min_psnr) / (max_psnr - min_psnr)
        return torch.clamp(normalized_psnr, min=0.0, max=1.0)
