# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Perturbations module for OOD detection.

Analyze how the likelihood of a normalizing flow responds to various
image augmentations.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from infqm.normalizing_flow.eval import RealNVPEvaluator

plt.style.use("utils/figurestil.mplstyle")


class ImageAugmentationAnalyzer:
    """Analyze a normalizing flow.

    Analyze how the likelihood of a normalizing flow responds to various
    image augmentations across multiple images in a folder.
    """

    def __init__(self, model_id: str | None) -> None:
        """Initialize the analyzer.

        Args:
            model_id: Path to folder containing images

        Raises:
            ValueError: If no images are found in the folder
        """
        self.image_folder = Path("ood_data/images")
        if model_id:
            self.ood_detector = RealNVPEvaluator(f"network_logs/{model_id}")
        self.results = {}
        self.raw_results = []  # Store raw results for CSV
        self.image_names = []  # Store image filenames
        self.images = []
        self._load_images()

        if len(self.images) == 0:
            raise ValueError(f"No images found in {self.image_folder}")

    def _load_images(self) -> None:
        """Load all images from the folder."""
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

        for filepath in sorted(self.image_folder.iterdir()):
            if filepath.suffix.lower() in valid_extensions:
                try:
                    img = Image.open(filepath)
                    img_array = np.array(img)
                    self.images.append(img_array)
                    self.image_names.append(filepath.name)
                except (OSError, ValueError) as e:
                    print("Skipping image '%s': %s", filepath.name, e)

    def augment_darker(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Make image darker by multiplying pixel values.

        Args:
            image: Input image array
            factor: Darkness factor (0 = black, 1 = no change)

        Returns:
            Darkened image array
        """
        return np.clip(image * (1.0 - factor), 0, 255).astype(image.dtype)

    def augment_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Reduce contrast by factor. Factor < 1 reduces contrast, factor > 1
        increases it.

        Args:
            image: Input image array
            factor: Contrast factor (0 = gray, 1 = no change)

        Returns:
            Contrast-adjusted image array
        """
        mean = np.mean(image)
        return np.clip((image - mean) * factor + mean, 0, 255).astype(image.dtype)

    def augment_vignette(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """Add vignette effect (darker edges). Intensity 0 = no effect, 1 =
        strong vignette.

        Args:
            image: Input image array
            intensity: Vignette intensity (0 to 1)

        Returns:
            Image array with vignette effect
        """
        h, w = image.shape[:2]
        y, x = np.ogrid[:h, :w]

        # Distance from center, normalized
        center_y, center_x = h / 2, w / 2
        max_dist = np.sqrt(center_y**2 + center_x**2)
        dist = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2) / max_dist

        # Create vignette mask (1 at center, decreases towards edges)
        vignette_mask = 1 - (dist * intensity)
        vignette_mask = np.clip(vignette_mask, 0, 1)

        # Apply to all channels if color image
        if len(image.shape) == 3:
            vignette_mask = vignette_mask[:, :, np.newaxis]

        return np.clip(image * vignette_mask, 0, 255).astype(image.dtype)

    def augment_gaussian_noise(self, image: np.ndarray, std: float) -> np.ndarray:
        """Add Gaussian noise to image.

        Args:
            image: Input image array
            std: Standard deviation of Gaussian noise

        Returns:
            Noisy image array
        """
        noise = np.random.normal(0, std, image.shape)
        noisy = image.astype(float) + noise
        return np.clip(noisy, 0, 255).astype(image.dtype)

    def augment_dirt(self, image: np.ndarray, num_spots: int) -> np.ndarray:
        """Simulate dirt on lens with random dark spots.

        Args:
            image: Input image array
            num_spots: Number of random dirt spots to add

        Returns:
            np.ndarray: Image array with simulated dirt spots
        """
        result = image.copy().astype(float)
        h, w = image.shape[:2]

        for _ in range(num_spots):
            # Random position and size
            cx = np.random.randint(0, w)
            cy = np.random.randint(0, h)
            radius = np.random.uniform(min(w, h) * 0.05, min(w, h) * 0.3)

            # Create circular spot
            y, x = np.ogrid[-cy : h - cy, -cx : w - cx]
            mask = x * x + y * y <= radius**2

            # Darken the spot with some transparency
            darkness = np.random.uniform(0.5, 0.1)
            result[mask] *= darkness

        return np.clip(result, 0, 255).astype(image.dtype)

    def augment_blur(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """Apply Gaussian blur to image.

        Args:
            image: Input image array
            sigma: Standard deviation for Gaussian kernel

        Returns:
            Blurred image array
        """
        if len(image.shape) == 3:
            # Apply blur to each channel
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:, :, c] = gaussian_filter(image[:, :, c], sigma=sigma)
            return result.astype(image.dtype)
        return gaussian_filter(image, sigma=sigma).astype(image.dtype)

    def augment_rolling_shutter(
        self, image: np.ndarray, shift: float, direction: str = "horizontal"
    ) -> np.ndarray:
        """Apply rolling shutter glitch effect to image.

        Args:
            image: Input image array
            shift: Maximum pixel shift amount (0 = no effect, higher = more distortion)
            direction: 'horizontal' or 'vertical' rolling shutter direction

        Returns:
            Image array with rolling shutter effect
        """
        if shift == 0:
            return image.astype(image.dtype)

        result = np.zeros_like(image)

        if direction == "horizontal":
            # Shift each row by an increasing amount
            for row in range(image.shape[0]):
                # Calculate shift for this row (linear progression)
                row_shift = int((row / image.shape[0]) * shift)
                if len(image.shape) == 3:
                    result[row, :, :] = np.roll(image[row, :, :], row_shift, axis=0)
                else:
                    result[row, :] = np.roll(image[row, :], row_shift)
        else:  # vertical
            # Shift each column by an increasing amount
            for col in range(image.shape[1]):
                col_shift = int((col / image.shape[1]) * shift)
                if len(image.shape) == 3:
                    result[:, col, :] = np.roll(image[:, col, :], col_shift, axis=0)
                else:
                    result[:, col] = np.roll(image[:, col], col_shift)

        return result.astype(image.dtype)

    def add_lens_flare(self, image, flare_center=None, intensity=0.5):
        """Adds a simple lens flare effect to the image.

        Args:
            image: Input image as a numpy array.
            flare_center: Tuple (x, y) for the center of the flare.
                If None, a random position is chosen.
            intensity: Float between 0 and 1 indicating the intensity of the flare.

        Returns:
            Image with lens flare effect applied.
        """
        h, w = image.shape[:2]
        overlay = np.zeros_like(image, dtype=np.float32)

        # Random if no position is given
        if flare_center is None:
            flare_center = (
                np.random.randint(int(w * 0.3), int(w * 0.7)),
                np.random.randint(int(h * 0.3), int(h * 0.7)),
            )

        # Mehrere konzentrische Licht-Halos
        num_flares = 6
        for i in range(num_flares):
            radius = np.random.randint(30, 200) * (i + 1)
            color = np.array(
                [255, 200 + np.random.randint(-50, 50), 100], dtype=np.float32
            )
            cv2.circle(overlay, flare_center, radius, color.tolist(), -1)
            overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=radius / 4)

        # Blend Overlay in Original image
        result = cv2.addWeighted(image.astype(np.float32), 1.0, overlay, intensity, 0)
        return np.clip(result, 0, 255).astype(np.uint8)

    def analyze_augmentation(
        self, aug_type: str, param_range: np.ndarray, num_samples: int = 10
    ) -> None:
        """Analyze how f(image) behaves under specific augmentation across all
        images.

        Args:
            aug_type: Type of augmentation
            param_range: Range of augmentation parameters to test
            num_samples: Number of random samples per parameter value per image

        Raises:
            ValueError: If aug_type is unknown
        """
        for param in param_range:
            # Iterate over all images
            for img_idx, image in enumerate(tqdm(self.images)):
                image_name = self.image_names[img_idx]

                for _ in range(num_samples):
                    # Apply augmentation
                    if aug_type == "darker":
                        aug_img = self.augment_darker(image, param)
                    elif aug_type == "noise":
                        aug_img = self.augment_gaussian_noise(image, param)
                    elif aug_type == "dirt":
                        aug_img = self.augment_dirt(image, int(param))
                    elif aug_type == "blur":
                        aug_img = self.augment_blur(image, param)
                    elif aug_type == "lens_flare":
                        aug_img = self.add_lens_flare(image, intensity=param)
                    elif aug_type == "rolling_shutter":
                        aug_img = self.augment_rolling_shutter(image, param)
                    else:
                        raise ValueError(f"Unknown augmentation type: {aug_type}")

                    # Evaluate function
                    _, result = self.ood_detector.test_single_image_features(aug_img)

                    # Store raw result
                    self.raw_results.append({
                        "image_name": image_name,
                        "augmentation_type": aug_type,
                        "augmentation_strength": param,
                        "f(image)": result,
                    })

        self.results[aug_type] = True  # Mark as completed

    def analyze_all(
        self,
        darker_range: np.ndarray | None = None,
        noise_range: np.ndarray | None = None,
        dirt_range: np.ndarray | None = None,
        blur_range: np.ndarray | None = None,
        lens_flare_range: np.ndarray | None = None,
        rolling_shutter_range: np.ndarray | None = None,
        num_samples: int = 10,
    ) -> None:
        """Analyze all augmentation types with default or custom ranges.

        Args:
            darker_range: Range of darkness factors (0 to 1)
            noise_range: Range of noise standard deviations
            dirt_range: Range of number of dirt spots
            blur_range: Range of blur sigma values
            lens_flare_range: Range of lens flare intensity (0 to 1)
            rolling_shutter_range: Range of rolling shutter shift values
            num_samples: Number of samples per parameter value
        """
        # Default ranges
        if darker_range is None:
            darker_range = np.linspace(0.0, 0.8, 10)
        if noise_range is None:
            noise_range = np.linspace(0, 10, 10)
        if dirt_range is None:
            dirt_range = np.linspace(0, 25, 10)
        if blur_range is None:
            blur_range = np.linspace(0, 2, 10)
        if lens_flare_range is None:
            lens_flare_range = np.linspace(0, 0.5, 10)
        if rolling_shutter_range is None:
            rolling_shutter_range = np.linspace(0, 2000, 10)

        self.analyze_augmentation("darker", darker_range, num_samples)

        print("Analyzing noise augmentation...")
        self.analyze_augmentation("noise", noise_range, num_samples)

        print("Analyzing dirt augmentation...")
        self.analyze_augmentation("dirt", dirt_range, num_samples)

        print("Analyzing blur augmentation...")
        self.analyze_augmentation("blur", blur_range, num_samples)

        print("Analyzing lens flare augmentation...")
        self.analyze_augmentation("lens_flare", lens_flare_range, num_samples)

        print("Analysing rolling shutter effect...")
        self.analyze_augmentation("rolling_shutter", rolling_shutter_range, num_samples)

    def save_results(self, filename) -> None:
        """Save all raw results to a single CSV file.

        Args:
            filename: Name of the CSV file to save results to
        """
        if not self.raw_results:
            return

        df = pd.DataFrame(self.raw_results)
        output_path = Path("ood_data/results") / filename

        df.to_csv(output_path, index=False)

    def plot_results(
        self,
        filename: str,
        separate_plots: bool = False,
        figsize: tuple[int, int] = (8, 4),
        csv_path: str | None = None,
    ) -> None:
        """Plot analysis results with 50% confidence interval.

        Can plot from raw_results or load from CSV.

        Args:
            separate_plots: If True, create separate plots for each augmentation
            figsize: Figure size
            filename: Path to save figure (optional)
            csv_path: Path to CSV file to load data from (optional)
        """
        save_path = Path("ood_data/results")
        # Load data from CSV if provided
        if csv_path:
            df_all = pd.read_csv(csv_path)
        else:
            if not self.raw_results:
                return
            df_all = pd.DataFrame(self.raw_results)

        # Get unique augmentation types
        aug_types = df_all["augmentation_type"].unique()

        # Compute statistics for each augmentation type
        plot_data = {}
        for aug_type in aug_types:
            df_aug = df_all[df_all["augmentation_type"] == aug_type]

            # Group by augmentation strength and compute statistics
            grouped = (
                df_aug
                .groupby("augmentation_strength")["f(image)"]
                .agg([
                    ("median", "median"),
                    ("std", "std"),
                    ("percentile_25", lambda x: np.percentile(x, 25)),
                    ("percentile_75", lambda x: np.percentile(x, 75)),
                    ("min", "min"),
                    ("max", "max"),
                ])
                .reset_index()
            )

            plot_data[aug_type] = grouped

        aug_labels = {
            "darker": "Darkness Factor",
            "noise": "Noise Std Dev",
            "dirt": "Number of Dirt Spots",
            "blur": "Blur Sigma",
            "lens_flare": "Lens Flare Intensity",
            "rolling_shutter": "Roling Shutter",
        }

        if separate_plots:
            # Create separate plots
            for aug_type, df in plot_data.items():
                _, ax = plt.subplots(figsize=(16, 9))

                ax.plot(
                    df["augmentation_strength"],
                    df["median"],
                    "o-",
                    linewidth=2,
                    markersize=6,
                    label="median",
                    color="#2E86AB",
                )
                ax.fill_between(
                    df["augmentation_strength"],
                    df["percentile_25"],
                    df["percentile_75"],
                    alpha=0.3,
                    label="50% Confidence Interval",
                    color="#2E86AB",
                )

                xlabel, ylabel = aug_labels.get(aug_type, "Parameter"), "Log-Likelihood"
                ax.set_xlabel(xlabel, fontsize=12)
                ax.set_ylabel(ylabel, fontsize=12)

                # Count unique images
                n_images = len(
                    df_all[df_all["augmentation_type"] == aug_type][
                        "image_name"
                    ].unique()
                )
                ax.set_title(
                    f"Effect of {aug_type.replace('_', ' ').title()} \
                    Augmentation\n({n_images} images)",
                    fontweight="bold",
                )
                ax.legend()
                ax.grid(True, alpha=0.3)
                path = save_path / (xlabel.replace(" ", "_") + "_" + filename)
                plt.savefig(path, dpi=300, bbox_inches="tight")

                plt.tight_layout()
                plt.show()
        else:
            save_path /= filename
            # Combined plot
            n_plots = len(plot_data)
            if n_plots <= 4:
                _, axes = plt.subplots(2, 2, figsize=figsize)
                axes = axes.flatten()
            else:
                _, axes = plt.subplots(2, 3, figsize=figsize)
                axes = axes.flatten()

            # colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"]
            colors = ["#385723", "#385723", "#385723", "#385723", "#385723", "#385723"]

            for idx, (aug_type, df) in enumerate(plot_data.items()):
                ax = axes[idx]
                color = colors[idx % len(colors)]

                ax.plot(
                    df["augmentation_strength"],
                    df["median"],
                    "o-",
                    linewidth=2,
                    markersize=6,
                    label="Median",
                    color=color,
                )
                ax.fill_between(
                    df["augmentation_strength"],
                    df["percentile_25"],
                    df["percentile_75"],
                    alpha=0.3,
                    label="50% CI",
                    color=color,
                )

                xlabel, ylabel = aug_labels.get(aug_type, "Parameter"), "Log-Likelihood"
                ax.set_xlabel(xlabel, fontsize=10)
                ax.set_ylabel(ylabel, fontsize=10)

                # Count unique images
                n_images = len(
                    df_all[df_all["augmentation_type"] == aug_type][
                        "image_name"
                    ].unique()
                )
                ax.set_title(
                    f"{aug_type.replace('_', ' ').title()}",
                    fontweight="bold",
                )
                ax.legend()
                ax.grid(True, alpha=0.3)

            # Hide unused subplots if any
            for idx in range(len(plot_data), len(axes)):
                axes[idx].set_visible(False)

            plt.tight_layout()

            plt.savefig(save_path, dpi=300, bbox_inches="tight")

    def save_augmented_sample(
        self, aug_type: str, param_value: float, output_name: str, image_idx: int = 0
    ) -> None:
        """Save an example of augmented image.

        Args:
            aug_type: Type of augmentation
            param_value: Parameter value for augmentation
            output_name: Path to save the image
            image_idx: Index of image to use from the loaded images

        Raises:
            ValueError: If image_idx is out of range or if aug_type is unknown
        """
        if image_idx >= len(self.images):
            raise ValueError(
                f"Image index {image_idx} out of range. Only {len(self.images)} "
                "images loaded."
            )

        original_image = self.images[image_idx]

        if aug_type == "darker":
            aug_img = self.augment_darker(original_image, param_value)
        elif aug_type == "noise":
            aug_img = self.augment_gaussian_noise(original_image, param_value)
        elif aug_type == "dirt":
            aug_img = self.augment_dirt(original_image, int(param_value))
        elif aug_type == "blur":
            aug_img = self.augment_blur(original_image, param_value)
        elif aug_type == "lens_flare":
            aug_img = self.add_lens_flare(original_image, intensity=param_value)
        elif aug_type == "rolling_shutter":
            aug_img = self.augment_rolling_shutter(original_image, param_value)
        else:
            raise ValueError(f"Unknown augmentation type: {aug_type}")

        # Save using matplotlib
        plt.figure(figsize=(10, 5))

        # plt.subplot(1, 2, 1)
        # plt.imshow(
        #     original_image, cmap="gray" if len(original_image.shape) == 2 else None
        # )
        # plt.title(f"Original Image {image_idx}")
        # plt.axis("off")

        # plt.subplot(1, 2, 2)
        plt.imshow(aug_img, cmap="gray" if len(aug_img.shape) == 2 else None)
        # plt.title(f'{aug_type.replace("_", " ").title()}: {param_value}')
        plt.axis("off")

        output_path = Path("data/results") / output_name
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()


def main():
    """Main function to run the analysis and plotting."""
    # # Create analyzer with image folder
    analyzer = ImageAugmentationAnalyzer("train_20260213_191852")
    # analyzer.analyze_augmentation("lens_flare", np.linspace(0.0,1.0,5), num_samples=1)

    # # Run all analyses
    # analyzer.analyze_all(num_samples=5)

    # # Save results to CSV
    # analyzer.save_results(filename='augmentation_results.csv')

    # Plot results (combined) with 50% confidence interval
    analyzer.plot_results(
        filename="non_seperate.jpg",
        separate_plots=False,
        csv_path=r"ood_data/results/augmentation_results.csv",
    )
    # analyzer.plot_results(filename="non_seperate.jpg", separate_plots=False)

    # # Plot results (separate) with 50% confidence interval
    # analyzer.plot_results(filename = "separate.jpg",separate_plots=True,
    #   csv_path = r"src/normalizing_flow/
    #           ood_detection/data/results/augmentation_results.csv")

    # Save sample augmented images (from first image in folder)
    # analyzer.save_augmented_sample('noise', 10.0, 'sample_noise.png', image_idx=0)
    # analyzer.save_augmented_sample(
    #   'rolling_shutter', 1000, 'sample_rolling_shutter.png', image_idx=0)
    # analyzer.save_augmented_sample('blur', 1.0, 'sample_blur.png', image_idx=0)
    # analyzer.save_augmented_sample("lens_flare", 0.6, "sample_cover.png", image_idx=0)
    # analyzer.save_augmented_sample('dirt', 1,'sample_cover.png', image_idx=0)
    # analyzer.save_augmented_sample('darker', 0.5, 'sample_darker.png', image_idx=0)


# Example usage
if __name__ == "__main__":
    main()
