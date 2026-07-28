# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Estimate Brightness of an image.

Brightness is the mean pixel value in grey-scaled image
"""

import cv2
import numpy as np
import torch

from infqm.base_metric import BaseMetric


class BrightnessMetric(BaseMetric):
    """Class to compute brightness of an image."""

    def calculate(self, cv_image):
        """Calculate average brightness of the image.

        Args:
            cv_image: OpenCV image (numpy array).

        Returns:
            Average brightness of the image (float between 0 and 1).
        """
        # Convert to grayscale if color image
        if len(cv_image.shape) == 3:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv_image

        return float(np.mean(gray)) / 255.0

    def calculate_tensor(self, image_batch):
        """Calculate average brightness of the image tensor.

        Args:
            image_batch: PyTorch tensor of shape (C, H, W) or (H, W).

        Returns:
            Average brightness of the image (float between 0 and 1).
        """
        gray = self.tensor_rgb2gray(image_batch=image_batch)

        return torch.mean(gray, dim=(1, 2))
