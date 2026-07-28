# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Load images on-the-fly from disk using PyTorch DataLoader."""

import os
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from infqm.datasets import MyDataset


class MyImages(Dataset):
    """Dataset that loads images on-the-fly."""

    def __init__(self, image_paths, transform=None) -> None:
        """Args:
        image_paths: List of image file paths
        transform: Optional transform to be applied on images.
        """
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        """Return the total number of images.

        Returns:
            int: Total number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, index) -> Image.Image:
        """Load and return an image at the specified index.

        Args:
            index: Index of the image to load

        Returns:
            Image.Image: Loaded image as a tensor.
        """
        img_path = self.image_paths[index]

        # Load image and convert to RGB
        image = Image.open(img_path).convert("RGB")

        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)

        return image


class ImageDataset:
    """Dataset that loads images on-the-fly."""

    def __init__(self, name) -> None:
        """Initialize dataset.

        Args:
            name: Name of Dataset
        """
        self.dataset = MyDataset(name)

    def collect_image_paths(self, folder_path) -> list[Path]:
        """Collect all image paths from a folder and subfolders.

        Args:
            folder_path: Path to the folder containing images (can have subfolders)

        Returns:
            List of image file paths found in the folder and subfolders.
        """
        folder_path = Path(folder_path)

        all_items = []
        for root, _, files in os.walk(folder_path):
            all_items.extend(os.path.join(root, file) for file in files)

        image_paths = []
        for file_path in tqdm(all_items, desc="Filtering images"):
            file_path = Path(file_path)
            if file_path.suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tiff",
                ".webp",
                ".gif",
            }:
                image_paths.append(file_path)

        return image_paths

    def create_image_dataloaders(
        self,
        random_state=42,
        batch_size=32,
        num_workers=16,
        image_size=None,
    ):
        """Create train and test DataLoaders with on-the-fly loading.

        Args:
            random_state: Random seed for reproducibility
            batch_size: Number of images per batch
            num_workers: Number of subprocesses for DataLoader
            image_size: Optional tuple (height, width) to resize images

        Returns:
            train_dataloader, val_dataloader, test_dataloader
        """
        # Define transforms
        transform = transforms.Compose([
            *([transforms.Resize(image_size)] if image_size is not None else []),
            transforms.ToTensor(),
        ])

        if self.dataset.has_split():
            # Load from separate folders
            train_paths = self.collect_image_paths(self.dataset.get_path("train"))
            test_paths = self.collect_image_paths(self.dataset.get_path("test"))
            val_paths = self.collect_image_paths(self.dataset.get_path("val"))
        else:
            # Split single folder
            all_paths = self.collect_image_paths(self.dataset.get_path("train"))

            tmp, test_paths = train_test_split(
                all_paths, test_size=0.2, random_state=random_state
            )
            train_paths, val_paths = train_test_split(
                tmp, test_size=0.1, random_state=random_state
            )

        # Create datasets
        train_dataset = MyImages(train_paths, transform=transform)
        val_dataset = MyImages(val_paths, transform=transform)
        test_dataset = MyImages(test_paths, transform=transform)

        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        return train_dataloader, val_dataloader, test_dataloader


def main():
    """Main function to create image dataloaders."""
    F = ImageDataset("bdd")
    train, _val, test = F.create_image_dataloaders()

    # Training loop
    for _epoch in range(1):
        for _images in tqdm(train, desc="Training"):
            pass  # Your training code
        for _images in tqdm(test, desc="Validation"):
            pass  # Your validation code


# Example usage:
if __name__ == "__main__":
    main()
