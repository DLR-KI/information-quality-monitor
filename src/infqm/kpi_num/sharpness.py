# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Description."""

import cv2
import torch
from numpy import pad

from infqm.base_metric import BaseMetric


class SharpnessMetric(BaseMetric):
    """Class to compute sharpness of an image using Laplacian variance."""

    def calculate(self, cv_image):
        """Calculate sharpness using Laplacian variance.

        Very fast edge-based sharpness measure

        Args:
            cv_image: OpenCV image (numpy array).

        Returns:
            float - normalized sharpness value
        """
        if len(cv_image.shape) == 3:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY) / 255.0
        else:
            gray = cv_image / 255.0

        # Compute Laplacian and return variance
        padded = pad(gray, pad_width=1, mode="edge")
        laplacian = (
            4 * gray
            - padded[:-2, 1:-1]  # top
            - padded[2:, 1:-1]  # bottom
            - padded[1:-1, :-2]  # left
            - padded[1:-1, 2:]  # right
        )
        variance = laplacian.var()

        # Normalize by typical range for 8-bit images
        return min(variance, 1.0)

    def calculate_tensor(self, image_batch):
        """Calculate sharpness using Laplacian variance for a batch of images.
        Very fast edge-based sharpness measure.

        Args:
            image_batch: Tensor of shape (batch_size, C, H, W) for color images
                        or (batch_size, 1, H, W) for grayscale images

        Returns:
            Tensor of shape (batch_size,) containing normalized sharpness for each image
        """
        gray = self.tensor_rgb2gray(image_batch=image_batch)

        # Laplacian kernel
        laplacian_kernel = torch.tensor(
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=gray.dtype, device=gray.device
        )

        # Add channel dimension to gray: (batch_size, H, W) -> (batch_size, 1, H, W)
        gray_4d = gray.unsqueeze(1)

        gray_4d_padded = torch.nn.functional.pad(
            gray_4d,
            pad=(1, 1, 1, 1),
            mode="replicate",  # (left, right, top, bottom)
        )

        # Add input and output channel dimensions to kernel: (3, 3) -> (1, 1, 3, 3)
        kernel_4d = laplacian_kernel.unsqueeze(0).unsqueeze(0)

        # Compute Laplacian using convolution
        # Output shape: (batch_size, 1, H, W)
        laplacian = torch.nn.functional.conv2d(gray_4d_padded, kernel_4d, padding=0)

        # Remove channel dimension: (batch_size, 1, H, W) -> (batch_size, H, W)
        laplacian = laplacian.squeeze(1)

        # Compute variance per image (over spatial dimensions)
        # Flatten spatial dimensions for each image
        laplacian_flat = laplacian.view(
            image_batch.shape[0], -1
        )  # Shape: (batch_size, H*W)
        variance = torch.var(laplacian_flat, dim=1)  # Shape: (batch_size,)

        return torch.clamp(variance, max=1.0)
