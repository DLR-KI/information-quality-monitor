# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Compute spectral centroid of an image using Fourier transform."""

import cv2
import numpy as np
import torch

from infqm.base_metric import BaseMetric


class SpectralCentroidMetric(BaseMetric):
    """Compute spectral centroid of an image using Fourier transform.

    Higher values indicate more high-frequency content (finer details).
    """

    def calculate(self, cv_image):
        """Calculate spectral centroid using Fourier transform.

        Args:
            cv_image: OpenCV image (numpy array).

        Returns:
            float - spectral centroid value (normalized to [0, 1], higher = more high
        """
        gray = (
            cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            if len(cv_image.shape) == 3
            else cv_image
        )

        # Compute 2D FFT
        f_shift = np.fft.fftshift(np.fft.fft2(gray))
        magnitude = np.abs(f_shift)

        # Create frequency coordinates
        rows, cols = gray.shape
        freq_rows = np.fft.fftshift(np.fft.fftfreq(rows))
        freq_cols = np.fft.fftshift(np.fft.fftfreq(cols))

        # Create 2D frequency magnitude grid
        freq_row_grid, freq_col_grid = np.meshgrid(freq_rows, freq_cols, indexing="ij")
        freq_magnitude = np.sqrt(freq_row_grid**2 + freq_col_grid**2)

        # Compute spectral centroid as weighted average of frequencies
        total_magnitude = np.sum(magnitude)
        if total_magnitude == 0:
            return 0.0

        spectral_centroid = np.sum(freq_magnitude * magnitude) / total_magnitude

        # Normalize to [0, 1] - max frequency magnitude is sqrt(0.5^2 + 0.5^2)
        max_freq = np.sqrt(0.5**2 + 0.5**2)

        return float(spectral_centroid / max_freq)

    def calculate_tensor(self, image_batch):
        """Calculate spectral centroid using Fourier transform.

        Args:
            image_batch: PyTorch tensor of shape (batch_size, C, H, W).

        Returns:
            tensor - spectral centroid value (normalized to [0, 1]
        """
        gray = self.tensor_rgb2gray(image_batch=image_batch)

        rows, cols = gray.shape[1], gray.shape[2]

        # Compute 2D FFT for each image in the batch
        f_transform = torch.fft.fft2(
            gray
        )  # Shape: (batch_size, H, W) with complex values
        f_shift = torch.fft.fftshift(f_transform, dim=(-2, -1))  # Shift per image
        magnitude = torch.abs(f_shift)  # Shape: (batch_size, H, W)

        # Create frequency coordinates (same for all images in batch)
        freq_rows = torch.fft.fftshift(torch.fft.fftfreq(rows, device=gray.device))
        freq_cols = torch.fft.fftshift(torch.fft.fftfreq(cols, device=gray.device))

        # Create 2D frequency magnitude grid
        freq_row_grid, freq_col_grid = torch.meshgrid(
            freq_rows, freq_cols, indexing="ij"
        )
        freq_magnitude = torch.sqrt(
            freq_row_grid**2 + freq_col_grid**2
        )  # Shape: (H, W)

        # Expand freq_magnitude to match batch dimension for broadcasting
        freq_magnitude = freq_magnitude.unsqueeze(0)  # Shape: (1, H, W)

        # Compute spectral centroid as weighted average of frequencies per image
        total_magnitude = torch.sum(magnitude, dim=(-2, -1))  # Shape: (batch_size,)

        # Compute weighted sum per image
        weighted_sum = torch.sum(
            freq_magnitude * magnitude, dim=(-2, -1)
        )  # Shape: (batch_size,)

        # Avoid division by zero
        spectral_centroid = torch.where(
            total_magnitude == 0,
            torch.zeros_like(total_magnitude),
            weighted_sum / total_magnitude,
        )

        # Normalize to [0, 1] - max frequency magnitude is sqrt(0.5^2 + 0.5^2)
        max_freq = torch.sqrt(torch.tensor(0.5**2 + 0.5**2, device=gray.device))
        return spectral_centroid / max_freq
