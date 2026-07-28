# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""eval.py: Class for evaluating normalizing flows on data."""

import importlib
import inspect
import os
import sys
from pathlib import Path

import numpy as np
import torch
from cv2 import imread

from infqm.base_metric import BaseMetric


class RealNVPEvaluator:
    """Class for evaluating normalizing flows on data."""

    def __init__(self, model_dir: str) -> None:
        """Load the trained model and feature scaler from directory.

        Args:
            model_dir: Directory containing the trained model and scaler files
        """
        self.load_model(model_dir=model_dir)
        self.metrics = []

    def load_model(self, model_dir) -> None:
        """Loads Model from Directory.

        Args:
            model_dir: Directory containing the trained model and scaler files

        Raises:
            FileNotFoundError: If the model file is not found
            ImportError: If the training script cannot be imported
        """
        training_script_name = "_script"

        # Add model directory to Python path to import the training script
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        try:
            # Import the classes from the training script
            training_module = __import__(training_script_name)
            real_nvp = training_module.RealNVP
        except ImportError as e:
            raise ImportError(
                f"Could not import from {training_script_name}.py in {model_dir}: {e}"
            ) from e

        # Load model
        model_path = os.path.join(model_dir, "normalizing_flow_model.pth")

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu")

        self.mode = checkpoint["mode"]

        # Recreate model with same architecture
        model = real_nvp(
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"],
            mode=self.mode,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        self.model = model

    def test_single_point(self, data_point: list[float], threshold: float = -10.0):
        """Test if a single point is OOD.

        Args:
            data_point: List of feature values
            threshold: Log probability threshold (lower = more strict)

        Returns:
            is_ood: Boolean indicating if point is OOD
            log_prob: Log probability of the point
        """
        # Convert to numpy array and reshape
        point = np.array(data_point).reshape(1, -1)

        # Convert to tensor
        point_tensor = torch.FloatTensor(point)

        # Compute log probability
        with torch.no_grad():
            log_prob = self.model.log_prob(point_tensor).item()

        # Determine if OOD based on threshold
        is_ood = log_prob < threshold

        return is_ood, log_prob

    def _load_features(self) -> None:
        """Dynamically load all metric classes from the metrics directory."""
        # Get the directory where this script is located
        metrics_dir = Path("src/infqm/kpi_num")

        # Create metrics directory if it doesn't exist
        metrics_dir.mkdir(exist_ok=True)

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
                        self.metrics.append(metric_instance)
            except ImportError:
                pass
            except TypeError:
                pass
            except AttributeError:
                pass

        self.metrics.sort(key=lambda m: m.get_name())

    def test_single_image_features(self, image):
        """Test quality of a single image based on extracted features.

        Args:
            image: Path to the image file or already loaded image array

        Returns:
            Log-likelihood of the image under the model based on its features.
        """
        assert self.mode == "features"

        if len(self.metrics) == 0:
            self._load_features()

        if isinstance(image, str):
            image = imread(image)
        # else image is already loaded

        features = [metric.calculate(image) for metric in self.metrics]

        return self.test_single_point(features)
