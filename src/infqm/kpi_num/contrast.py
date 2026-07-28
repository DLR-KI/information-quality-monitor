# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Estimate Image Contrast.

Contrast is defined by standard deviation in greyscale values.
"""

import cv2
import numpy as np
import torch

from infqm.base_metric import BaseMetric


class ContrastMetric(BaseMetric):
    """Class to compute contrast of an image."""

    def calculate(self, cv_image):
        """Calculate contrast using standard deviation method.

        Args:
            cv_image: OpenCV image (numpy array).

        Returns:
            float - contrast value (higher = more contrast).
        """
        # Convert to grayscale if color image
        if len(cv_image.shape) == 3:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv_image

        return float(np.std(gray)) / 255.0

    def calculate_tensor(self, image_batch):
        """Calculate contrast using standard deviation method.

        Args:
            image_batch: PyTorch tensor of shape (C, H, W) or (H, W)

        Returns:
            tensor - contrast value (higher = more contrast).
        """
        gray = self.tensor_rgb2gray(image_batch=image_batch)

        return torch.std(gray, dim=(1, 2))
