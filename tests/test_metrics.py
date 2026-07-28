# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
import importlib
import inspect
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from infqm.base_metric import BaseMetric


@pytest.fixture
def all_image_paths():
    """Fixture to get all image paths from test_images folder."""
    # Get all common image formats
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp"]
    image_paths = []
    test_images_folder = Path("ood_data/images")

    for ext in image_extensions:
        image_paths.extend(test_images_folder.glob(ext))
        # Also check uppercase extensions
        image_paths.extend(test_images_folder.glob(ext.upper()))

    if not image_paths:
        pytest.skip("No images found in test_images folder")

    return image_paths


@pytest.fixture
def all_metrics():
    metrics_dir = Path("src") / "infqm" / "kpi_num"

    # Add metrics directory to Python path
    if str(metrics_dir) not in sys.path:
        sys.path.insert(0, str(metrics_dir))

    # Find all Python files in the metrics directory
    metric_files = []
    if metrics_dir.exists():
        metric_files.extend(
            file_path.stem
            for file_path in metrics_dir.glob("*.py")
            if file_path.name != "__init__.py"
        )

    metrics = []

    # Load each metric module and instantiate metric classes
    for module_name in metric_files:
        try:
            # Import the module
            module = importlib.import_module(module_name)

            # Find all classes in the module that inherit from BaseMetric
            for _, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, BaseMetric)
                    and obj != BaseMetric
                ):
                    # Instantiate the metric class
                    metric_instance = obj()
                    metrics.append(metric_instance)
        except (ImportError, AttributeError) as e:
            print("Failed to load metric module '%s': %s", module_name, e)
            metrics.sort(key=lambda m: m.get_name())

    return metrics


@pytest.mark.parametrize("i", range(12))
def test_brightness_metric_consistency(i, all_metrics, all_image_paths) -> None:
    """Test that calculate and calculate_tensor produce the same results on
    real images."""
    metric = all_metrics[i]

    for image_path in all_image_paths:
        # Load image with OpenCV (BGR format)
        cv_image = cv2.imread(str(image_path))

        if cv_image is None:
            pytest.fail(f"Failed to load image: {image_path}")

        # Calculate using OpenCV method
        brightness_cv = metric.calculate(cv_image)

        # Convert BGR to RGB and normalize to [0, 1] for PyTorch
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        tensor_image = torch.from_numpy(rgb_image).float() / 255.0

        # Reshape to (B, C, H, W) format for batch processing
        tensor_image = tensor_image.permute(2, 0, 1).unsqueeze(0)

        # Calculate using tensor method
        brightness_tensor = metric.calculate_tensor(tensor_image)
        brightness_tensor = brightness_tensor.item()
        # # Assert they are approximately equal
        assert np.isclose(brightness_cv, brightness_tensor, rtol=1e-2, atol=1e-3), (
            f"Metric {metric.get_name()}: Mismatch for {image_path.name}: CV={brightness_cv:.6f}, Tensor={brightness_tensor:.6f}"
        )


@pytest.mark.parametrize("i", range(12))
def test_brightness_metric_batch(i, all_metrics, all_image_paths) -> None:
    """Test that calculate_tensor works correctly with batches of real
    images."""
    metric = all_metrics[i]

    # Load all images
    cv_images = []
    for image_path in all_image_paths:
        cv_image = cv2.imread(str(image_path))
        if cv_image is not None:
            cv_images.append(cv_image)

    if len(cv_images) < 2:
        pytest.skip("Need at least 2 images for batch testing")

    # Take first few images (limit batch size for testing)
    cv_images = cv_images[:5]

    # Calculate individual brightnesses
    brightnesses_cv = [metric.calculate(img) for img in cv_images]

    # Find common size for batching (resize all to same dimensions)
    # Use the size of the first image
    target_size = (cv_images[0].shape[1], cv_images[0].shape[0])  # (width, height)
    cv_images_resized = [cv2.resize(img, target_size) for img in cv_images]

    # Recalculate CV brightnesses on resized images
    brightnesses_cv = [metric.calculate(img) for img in cv_images_resized]

    # Convert to batch tensor
    rgb_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in cv_images_resized]
    tensor_batch = torch.stack([
        torch.from_numpy(img).float().permute(2, 0, 1) / 255.0 for img in rgb_images
    ])

    # Calculate batch brightnesses
    brightnesses_tensor = metric.calculate_tensor(tensor_batch).tolist()

    # Compare each result
    for _, (cv_bright, tensor_bright) in enumerate(
        zip(brightnesses_cv, brightnesses_tensor, strict=False)
    ):
        assert np.isclose(cv_bright, tensor_bright, rtol=1e-2, atol=1e-3), (
            f"Batch item {metric.get_name()}: CV={cv_bright:.6f}, Tensor={tensor_bright:.6f}"
        )


if __name__ == "__main__":
    am = all_metrics()
    ap = all_image_paths()
    for i in [7]:
        # test_brightness_metric_batch(i, am, ap)
        test_brightness_metric_consistency(i, am, ap)
