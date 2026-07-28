# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Description."""

import cv2
import numpy as np
import torch

from infqm.base_metric import BaseMetric


class EnergyMetric(BaseMetric):
    """Class to compute energy of an image using Fourier transform."""

    def calculate(self, cv_image):
        """Calculate image energy using Fourier transform.

        Args:
            cv_image: OpenCV image (numpy array).

        Returns:
            float - energy of the image (higher = more energy).
        """
        gray = (
            cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY) / 255.0
            if len(cv_image.shape) == 3
            else cv_image / 255.0
        )

        # Compute 2D FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        # Energy is sum of squared magnitudes, normalized
        energy = np.sum(magnitude**2)
        return float(energy) / (gray.shape[0] * gray.shape[1]) ** 2

    def calculate_tensor(self, image_batch):
        """Calculate image energy using Fourier transform (tensor version).

        Args:
            image_batch: PyTorch tensor of shape (batch_size, C, H, W).

        Returns:
            Tensor of shape (batch_size,) containing energy of each image.
        """
        gray = self.tensor_rgb2gray(image_batch=image_batch)

        # Compute 2D FFT
        f_transform = torch.fft.fft2(
            gray
        )  # Shape: (batch_size, H, W) with complex values
        f_shift = torch.fft.fftshift(f_transform, dim=(-2, -1))  # Shift per image
        magnitude = torch.abs(f_shift)  # Shape: (batch_size, H, W)

        # Energy is sum of squared magnitudes per image, normalized
        # by spatial dimensions
        energy = torch.sum(
            magnitude**2, dim=(-2, -1)
        )  # Sum over H and W, shape: (batch_size,)

        # Normalize by number of pixels (H * W)
        height, width = gray.shape[1], gray.shape[2]
        return energy / ((height * width) ** 2)
