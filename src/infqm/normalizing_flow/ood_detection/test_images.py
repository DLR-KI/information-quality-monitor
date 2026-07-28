# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Test images for OOD detection."""

import csv
import os
import pathlib

from tqdm import tqdm

from infqm.datasets import MyDataset
from infqm.normalizing_flow.eval import RealNVPEvaluator


class Testing:
    """Class to test images for OOD detection."""

    def __init__(self, model) -> None:
        """Initialize OOD detector with given model.

        Args:
            model: Name of the model directory to load (e.g., "train_20251013_090123").
        """
        model_dir = (
            "network_logs/" + model
        )  # Directory containing model and scaler files
        self.ood_detector = RealNVPEvaluator(model_dir)
        self.name = model[6:]

    def test_images(self, image: str):
        """Test a single image for OOD detection and return log-likelihood.

        Args:
            image: Filename of the image to test.

        Returns:
            Log-likelihood of the image under the model
        """
        image = "ood_data/images/" + image

        _, log_prob = self.ood_detector.test_single_image_features(image)

        return log_prob

    def test_dataset(self, dataset: MyDataset) -> None:
        """Test all images in a dataset for OOD detection, save to CSV.

        Args:
            dataset: MyDataset object containing paths to the dataset splits

        Returns:
            None (results are saved to a CSV file)
        """
        all_image_paths = []
        path = (
            dataset.get_path("test")
            if dataset.has_split()
            else dataset.get_path("train")
        )
        for root, _, files in os.walk(path):
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

        data = []

        for img_path in tqdm(all_image_paths, desc="Processing images"):
            _, log_likelihood = self.ood_detector.test_single_image_features(img_path)
            item = {"file": img_path, "log_likelihood": log_likelihood}
            data.append(item)

        fieldnames = data[0].keys()

        csv_path = f"ood_data/results/OOD_full_{self.name}_{dataset.get_name()}.csv"

        # Write to CSV
        with pathlib.Path(csv_path).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)


def main():
    """Main function to run OOD tests on specified images and datasets."""
    T = Testing(model="train_20251013_090123")
    # T.test_images(image = "ba147ca3-583de3f4.jpg")
    # T.test_images(image = "cabc9045-5a50690f.jpg")
    # T.test_images(image = "cac07407-76e4c968.jpg")

    # T.test_images(image="sample_lens_flare.png")

    # print("-------")

    # dataset = MyDataset("carla")
    # T.test_dataset(dataset)

    # dataset = MyDataset("zod")
    # T.test_dataset(dataset)

    given_dataset = MyDataset("bdd")
    T.test_dataset(given_dataset)


if __name__ == "__main__":
    main()
