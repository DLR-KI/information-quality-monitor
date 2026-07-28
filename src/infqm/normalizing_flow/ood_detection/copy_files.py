# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Copy-files module for OOD detection.

Utility to copy a random selection of images from one folder to another
for OOD testing.
"""

import os
import random
import shutil
from pathlib import Path


def copy_random_images(source_folder, destination_folder, num_images=50) -> None:
    """Copy a specified number of random images from source to destination
    folder.

    Args:
        source_folder (str): Path to the folder containing the original images.
        destination_folder (str): Destination.
        num_images (int): Number of random images to copy (default is 50).
    """
    # supported image extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

    # create destination folder if it doesn't exist
    Path(destination_folder).mkdir(exist_ok=True, parents=True)

    # collect all image files from the source folder
    image_files = [
        file
        for file in os.listdir(source_folder)
        if Path(file).suffix.lower() in image_extensions
    ]

    # check if there are enough images
    total_images = len(image_files)

    if total_images == 0:
        return

    # adjust number if fewer images are available
    num_to_copy = min(num_images, total_images)

    # select random images
    selected_images = random.sample(image_files, num_to_copy)

    # copy the images
    copied = 0
    for img_file in selected_images:
        source_path = os.path.join(source_folder, img_file)
        dest_path = os.path.join(destination_folder, img_file)

        try:
            shutil.copy2(source_path, dest_path)
            copied += 1
        except FileNotFoundError:
            pass
        except PermissionError:
            pass
        except OSError:
            pass
