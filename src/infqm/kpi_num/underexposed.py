# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Ration of very low pixel values compared to the rest."""

import cv2
import numpy as np

from infqm.base_metric import BaseMetric


class UnderExposed(BaseMetric):
    """Class to compute underexposure of an image."""

    def get_exposure_limit(self) -> int:
        """Get Limit for under exposed images.

        Returns:
            int - pixel value below which is considered underexposed.
        """
        return 16

    def calculate(self, cv_image):
        """Calculate exposure quality using histogram analysis.

        Fast method based on pixel distribution.

        Args:
            cv_image: OpenCV image (numpy array).

        Returns:
            float - exposure quality (0-1, higher is better)
        """
        if len(cv_image.shape) == 3:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv_image

        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize

        # Penalize overexposure (too many whites) and underexposure (too many blacks)
        return np.sum(hist[: self.get_exposure_limit()])  # Very dark pixels

    def calculate_tensor(self, image_batch):
        """Calculate exposure quality using histogram analysis Fast method
        based on pixel distribution.

        Args:
            image_batch: PyTorch tensor of shape (C, H, W) or (H, W).

        Returns:
            tensor - exposure quality (0-1, higher is better).
        """
        gray = self.tensor_rgb2gray(image_batch=image_batch) * 255.0
        gray = gray.round()
        gray_flat = gray.view(gray.shape[0], -1)

        # Count pixels with brightness >= 240 for each image (overexposed pixels)
        overexposed_pixels = (
            (gray_flat < self.get_exposure_limit()).sum(dim=1).float()
        )  # Shape: (batch_size,)

        # Normalize by total number of pixels to get proportion
        total_pixels = gray_flat.shape[1]
        return overexposed_pixels / total_pixels
