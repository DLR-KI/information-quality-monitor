# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Generate training data for normalizing flow."""

import csv
import importlib
import inspect
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from cv2 import imread
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

from infqm.base_metric import BaseMetric
from infqm.datasets import MyDataset


class FeatureLoader:
    """Creates/Loads/Processes Features of a dataset."""

    def __init__(self, dataset_name) -> None:
        """Choose the wanted dataset.

        Args:
            dataset_name: Name of the dataset to load.
        """
        self.dataset_name = dataset_name
        self.random = dataset_name == "random"

        self.dataset = MyDataset(dataset_name) if not self.random else MyDataset()

        self.metrics = []

    def get_feature_path(self, dataset_type: str = "train") -> Path:
        """Get path to CSV file for given dataset type (train/test/val).

        Args:
            dataset_type: Type of dataset split (e.g., "train", "test", "val")

        Returns:
            Path to the CSV file where features for the given dataset type are stored.
        """
        prefix = Path("features")

        return prefix / f"{self.dataset_name}_{dataset_type}.csv"

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

    def create_features(self, dataset_types=None, handcraftet=True) -> None:
        """Create CSV with image properties from all images.

        Args:
            dataset_types: List of dataset splits to process
            handcraftet: Whether to use handcrafted metrics or latent features
        """
        self._load_features()
        if not handcraftet:
            model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            latent_embedding = nn.Sequential(*list(model.children())[:-1]).to("cuda")
        else:
            latent_embedding = None

        if not dataset_types:
            if self.dataset.has_split():
                dataset_types = ["train", "test", "val"]
            else:
                dataset_types = ["train"]
        # Collect all image paths
        for dataset_type in dataset_types:
            all_image_paths = []

            for root, _, files in os.walk(self.dataset.get_path(dataset_type)):
                all_image_paths.extend(
                    os.path.join(root, file)
                    for file in files
                    if file.lower().endswith((
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".bmp",
                        ".tiff",
                        ".gif",
                    ))
                )

            # Will store all rows here
            rows = []
            all_metric_names = {metric.get_name() for metric in self.metrics}

            for img_path in tqdm(all_image_paths, desc="Processing images"):
                filename = Path(img_path).name
                image = imread(img_path)
                if handcraftet:
                    results = {
                        metric.get_name(): metric.calculate(image)
                        for metric in self.metrics
                    }
                else:
                    image_tensor = (
                        torch
                        .from_numpy(image)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .float()
                        .to("cuda")
                    )
                    with torch.no_grad():
                        features = latent_embedding(image_tensor)
                        features = features.squeeze()

                    results = {
                        f"latent_{i + 1}": features[i].item()
                        for i in range(len(features))
                    }

                row = {"filename": filename}
                row.update(results)
                rows.append(row)

            if handcraftet:
                with Path(self.get_feature_path(dataset_type)).open(
                    "w", newline="", encoding="utf-8"
                ) as csvfile:
                    writer = csv.DictWriter(
                        csvfile, fieldnames=["filename", *sorted(all_metric_names)]
                    )
                    writer.writeheader()

                    for row in rows:
                        writer.writerow(row)

            else:
                dataset_type += "_latent"
                with Path(self.get_feature_path(dataset_type)).open(
                    "w", newline="", encoding="utf-8"
                ) as csvfile:
                    writer = csv.DictWriter(
                        csvfile,
                        fieldnames=["filename"]
                        + [f"latent_{i + 1}" for i in range(512)],
                    )
                    writer.writeheader()

                    for row in rows:
                        writer.writerow(row)

    def create_feature_dataloaders(
        self, random_state: int = 42, batch_size: int = 32
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Create dataloaders for the generated features.

        Args:
            random_state: Random seed for reproducibility when splitting data
            batch_size: Batch size for the dataloaders

        Returns:
            Tuple of (train_loader, val_loader, test_loader) DataLoader objects.
        """
        if self.dataset.has_split():
            x_train = pd.read_csv(self.get_feature_path("train"))
            x_train = x_train.drop(columns=["filename"]).values

            x_val = pd.read_csv(self.get_feature_path("val"))
            x_val = x_val.drop(columns=["filename"]).values

            x_test = pd.read_csv(self.get_feature_path("test"))
            x_test = x_test.drop(columns=["filename"]).values

        else:
            df = pd.read_csv(self.get_feature_path("train"))

            # Drop the 'filename' column
            df = df.drop(columns=["filename"])

            # First split: separate out test set (e.g., 20% test)
            x_temp, x_test = train_test_split(
                df.values, test_size=0.2, random_state=random_state
            )
            # Second split: split remaining data into train and val
            # (e.g., 80% train, 20% val of temp)
            # This gives final split of approximately 64% train, 16% val, 20% test
            x_train, x_val = train_test_split(
                x_temp, test_size=0.1, random_state=random_state
            )

            # Convert back to tensors and move to device
        x_train = torch.FloatTensor(x_train)
        x_val = torch.FloatTensor(x_val)
        x_test = torch.FloatTensor(x_test)

        # Create TensorDataset objects
        train_dataset = TensorDataset(x_train)
        val_dataset = TensorDataset(x_val)
        test_dataset = TensorDataset(x_test)

        # Create DataLoader objects
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )

        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )

        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )
        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    Z = FeatureLoader("my_dataset")
    Z.create_features()
