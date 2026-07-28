# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Class for handling image datasets."""

from pathlib import Path

import yaml


class Dataset:
    """Class for supervised learning datasets."""

    def __init__(self, data: str | None = None, labels: str | None = None) -> None:
        """Initialize dataset with data and labels paths.

        Args:
            data: Path to dataset data.
            labels: Path to dataset labels (optional).
        """
        self.data = data
        self.labels = labels

    def __str__(self) -> str:
        """String representation of dataset.

        Returns:
            String describing the dataset with data and labels paths.
        """
        return f"Dataset(data={self.data}, labels={self.labels})"

    def get_data(self):
        """Get path to dataset data.

        Returns:
            Path to dataset data.
        """
        return self.data

    def get_labels(self):
        """Get path to dataset labels.

        Returns:
            Path to dataset labels.
        """
        return self.labels


class MyDataset:
    """Class for handling image datasets."""

    def __init__(self, name: str | None = None) -> None:
        """Load yaml file corresponding to dataset name.

        Args:
            name: Name of dataset (optional)
        """
        if name:
            with (Path("datasets") / f"{name}.yaml").open(encoding="utf-8") as stream:
                try:
                    dataset = yaml.safe_load(stream)
                    self.name = dataset.get("name", None)
                    split_tmp = dataset.get("test", None)
                    train_data = dataset["train"]["data"]
                    test_labels = dataset["test"]["labels"]
                    train_labels = dataset["train"]["labels"]
                    val_labels = dataset["val"]["labels"]
                    test_data = ""
                    val_data = ""

                    if split_tmp.get("data", None) is not None:
                        self.split = dataset["test"]["data"] is not None
                        test_data = str(dataset["test"]["data"])
                        val_data = str(dataset["val"]["data"])
                    else:
                        self.split = False
                        test_data = None
                        val_data = None

                    self.train_dataset = Dataset(data=train_data, labels=train_labels)
                    self.test_dataset = Dataset(data=test_data, labels=test_labels)
                    self.val_dataset = Dataset(data=val_data, labels=val_labels)
                except yaml.YAMLError:
                    pass
        else:
            self.split = True

    def __str__(self) -> str:
        """String representation of dataset.

        Returns:
            String describing the dataset.
        """
        return self.name

    def get_name(self) -> str | None:
        """Get name of dataset.

        Returns:
            Name of dataset.
        """
        return self.name

    def has_split(self) -> bool:
        """Check if dataset has split.

        Returns:
            True if dataset has split, False otherwise.
        """
        return self.split

    def get_path(self, typ_of_set: str) -> Path | None:
        """Get path to dataset split.

        Args:
            typ_of_set: Type of dataset split (train, test, val)

        Returns:
            Path to dataset split or None if not available.
        """
        assert typ_of_set in {"train", "test", "val"}
        if typ_of_set == "train":
            return self.train_dataset.get_data()
        if typ_of_set == "test":
            return self.test_dataset.get_data()
        return self.val_dataset.get_data()
