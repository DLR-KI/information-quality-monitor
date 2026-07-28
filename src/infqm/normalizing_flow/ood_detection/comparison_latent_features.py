# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Evaluate latent features for OOD detection."""

import torch
from cv2 import imread
from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights

from infqm.normalizing_flow.eval import RealNVPEvaluator
from infqm.normalizing_flow.ood_detection.perturbations import ImageAugmentationAnalyzer


class LatentEmbedding(RealNVPEvaluator):
    """Latent OOD Detector.

    Evaluate OOD detection using latent features from a pretrained
    ResNet18.
    """

    def __init__(self, model_id) -> None:
        """Initialize with given model ID.

        Initialize with given model ID and set up ResNet18 for feature
        extraction.

        Args:
            model_id: Identifier for the model to load
        """
        super().__init__(model_id)
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.latent_embedding = nn.Sequential(*list(model.children())[:-1]).to("cuda")

    def to(self, device) -> None:
        """Move model to specified device.

        Args:
            device: Device to move the model to (e.g., "cuda" or "cpu")
        """
        self.model.to(device)
        self.device = device

    def test_single_image_features(self, image):
        """Test a single image for OOD detection using its latent features.

        Args:
            image: Path to the image file or already loaded image array

        Returns:
            Log-likelihood of the image under the model based on its latent features.
        """
        if isinstance(image, str):
            image = imread(image)
        # else image is already loaded

        image_tensor = (
            torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to("cuda")
        )
        with torch.no_grad():
            features = self.latent_embedding(image_tensor)
            features = features.squeeze()
        return self.test_single_point(features)


class LatentSpaceAugmentation(ImageAugmentationAnalyzer):
    """Analyze how augmentations affect latent features for OOD detection."""

    def __init__(self, model_id) -> None:
        """Initialize with given model ID.

        Initialize with given model ID and set up latent embedding for
        analysis.

        Args:
            model_id: Identifier for the model to load
        """
        super().__init__(model_id=None)
        self.ood_detector = LatentEmbedding(f"network_logs/{model_id}")

    def __str__(self) -> str:
        return "LatentSpaceAugmentation using model: " + str(self.ood_detector.model)

    def __repr__(self) -> str:
        return "LatentSpaceAugmentation using model: " + str(self.ood_detector.model)


def main():
    """Main function to run the latent space augmentation analysis."""
    analyzer = LatentSpaceAugmentation("train_20260213_191852")
    # analyzer.analyze_all(num_samples=1)
    # analyzer.save_results(filename='nf_latent_space_results.csv')
    analyzer.plot_results(
        filename="nf_non_seperate.pdf",
        separate_plots=False,
        csv_path=r"ood_data/results/nf_latent_space_results.csv",
    )


if __name__ == "__main__":
    main()
