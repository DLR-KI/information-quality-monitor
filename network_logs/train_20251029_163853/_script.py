# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Description."""

import importlib
import inspect
import sys
from pathlib import Path

import numpy as np
import torch
from logger import Register
from torch import nn, optim
from tqdm import tqdm

from infqm.base_metric import BaseMetric
from infqm.normalizing_flow.generate_training_data import FeatureLoader
from infqm.normalizing_flow.image_loader import ImageDataset

NUM_EPOCHS = 200
BATCH_SIZE = 64

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


def compute_normalization_stats(train_loader, device):
    """Compute min and max values for each feature across the entire training
    set.

    Args:
        train_loader: DataLoader for training data
        device: torch device

    Returns:
        feature_min: Tensor of minimum values per feature
        feature_max: Tensor of maximum values per feature
    """
    feature_min = None
    feature_max = None

    for (batch,) in tqdm(train_loader, desc="Computing stats"):
        batch = batch.to(device)

        if feature_min is None:
            feature_min = batch.min(dim=0)[0]
            feature_max = batch.max(dim=0)[0]
        else:
            feature_min = torch.min(feature_min, batch.min(dim=0)[0])
            feature_max = torch.max(feature_max, batch.max(dim=0)[0])

    return feature_min, feature_max


class CouplingLayer(nn.Module):
    """Single coupling layer for Real NVP."""

    def __init__(self, input_dim: int, hidden_dim: int, mask: torch.Tensor) -> None:
        """Initialize coupling layer.

        Args:
            input_dim: Dimensionality of input features
            hidden_dim: Number of hidden units in scale/translate networks
            mask: Binary mask to split input features.
        """
        super().__init__()
        self.register_buffer("mask", mask)
        self.scale_net = self._create_network(input_dim, hidden_dim)
        self.translate_net = self._create_network(input_dim, hidden_dim)

    def _create_network(self, input_dim: int, hidden_dim: int) -> nn.Module:
        """Create a simple MLP for scale/translation.

        Args:
            input_dim: Dimensionality of input features
            hidden_dim: Number of hidden units in the network

        Returns:
            An nn.Module representing the MLP.
        """
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh(),  # Bounded output for stability
        )

    def forward(
        self, x: torch.Tensor, inverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward or inverse pass through coupling layer.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            inverse: Whether to perform inverse transformation

        Returns:
            y: Transformed tensor of shape (batch_size, input_dim)
            log_det_j: Log-determinant of the Jacobian for this transformation
        """
        masked_x = x * self.mask
        s = self.scale_net(masked_x)
        t = self.translate_net(masked_x)

        if not inverse:
            # Forward transformation
            y = x * torch.exp(s * (1 - self.mask)) + t * (1 - self.mask)
            log_det_j = torch.sum(s * (1 - self.mask), dim=1)
        else:
            # Inverse transformation
            y = (x - t * (1 - self.mask)) * torch.exp(-s * (1 - self.mask))
            log_det_j = -torch.sum(s * (1 - self.mask), dim=1)

        return y, log_det_j


class RealNVP(nn.Module):
    """Real NVP normalizing flow implementation."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 6,
        mode: str = "images",
        feature_min=None,
        feature_max=None,
    ) -> None:
        """Initialize Real NVP model.

        Args:
            hidden_dim: Number of hidden units in coupling layers
            num_layers: Number of coupling layers
            mode: Whether to compute features from "images" or use "features" directly
            feature_min: Optional tensor of minimum values for feature normalization
            feature_max: Optional tensor of maximum values for feature normalization.
        """
        super().__init__()
        self.mode = mode

        self.image_quality_dimensions = []
        self.image_quality_dimensions_path = Path.cwd() / "src" / "kpi_num"

        self._load_image_quality_dimensions()
        self.image_quality_dimensions = sorted(
            self.image_quality_dimensions, key=lambda x: x.get_name()
        )

        self.input_dim = len(self.image_quality_dimensions)
        self.input_names = [iq.name for iq in self.image_quality_dimensions]

        self.num_layers = num_layers

        # Create coupling layers
        self.coupling_layers = nn.ModuleList()

        for i in range(num_layers):
            # Alternating mask pattern
            mask = torch.zeros(self.input_dim)

            if i % 2 == 0:
                mask[: self.input_dim // 2] = 1
            else:
                mask[self.input_dim // 2 :] = 1

            coupling_layer = CouplingLayer(self.input_dim, hidden_dim, mask)
            self.coupling_layers.append(coupling_layer)

        if feature_min is not None and feature_max is not None:
            self.register_buffer("feature_min", feature_min)
            self.register_buffer("feature_max", feature_max)
        else:
            self.register_buffer("feature_min", torch.zeros(self.input_dim))
            self.register_buffer("feature_max", torch.ones(self.input_dim))

    def _load_image_quality_dimensions(self) -> None:
        """Load all available vectorized image metrics."""
        metrics_dir = self.image_quality_dimensions_path

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
                kpi_module = importlib.import_module(module_name)

                # Find all classes in the module that inherit from BaseMetric
                for _, obj in inspect.getmembers(kpi_module):
                    if (
                        inspect.isclass(obj)
                        and issubclass(obj, BaseMetric)
                        and obj != BaseMetric
                    ):
                        # Instantiate the metric class
                        quality_measurement_instance = obj()
                        self.image_quality_dimensions.append(
                            quality_measurement_instance
                        )

            except (ImportError, AttributeError, TypeError) as e:
                print("Skipping module '%s': %s", module_name, e)

    def compute_features(self, x: torch.Tensor):
        """Compute features from input tensor.

        Args:
            x: Input tensor (e.g., batch of images or precomputed features)

        Returns:
            Tensor of computed features for each input sample.

        Raises:
            RuntimeError: If mode is "images" but no metrics are loaded.
        """
        if self.mode == "images":
            if not self.image_quality_dimensions:
                raise RuntimeError("No metrics loaded. Cannot perform forward pass.")

            evaluations = []

            # Compute Quality Attributes
            for quality_dim in self.image_quality_dimensions:
                output = quality_dim.calculate_tensor(x)

                # Ensure output has batch dimension
                if output.dim() == 1 and len(self.image_quality_dimensions) > 1:
                    output = output.unsqueeze(0)

                elif output.dim() == 0:  # Scalar output
                    output = output.unsqueeze(0).unsqueeze(0)

                evaluations.append(output)

            return torch.cat(evaluations, dim=0).T

        return x

    def normalize_features(self, x):
        """Normalize features to [0, 1] range using min-max normalization.

        Args:
            x: Tensor of features to normalize

        Returns:
            Normalized features tensor.
        """
        # Avoid division by zero
        range_vals = self.feature_max - self.feature_min
        range_vals = torch.where(
            range_vals == 0, torch.ones_like(range_vals), range_vals
        )
        return (x - self.feature_min) / range_vals

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the flow.

        Args:
            x: Input tensor (e.g., batch of images or precomputed features)

        Returns:
            z: Transformed tensor in latent space
            log_det_j: Log-determinant of the Jacobian for the entire transformation.
        """
        plain_features = self.compute_features(x)
        features = self.normalize_features(plain_features)

        log_det_j = torch.zeros(features.size(0), device=features.device)
        z = features

        for layer in self.coupling_layers:
            z_new, ldj = layer(z, inverse=False)
            z = z_new
            log_det_j += ldj

        return z, log_det_j

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse transformation (for sampling).

        Args:
            z: Tensor in latent space to transform back to input space

        Returns:
            torch.Tensor: Reconstructed input tensor.
        """
        x = z

        for layer in reversed(self.coupling_layers):
            x_new, _ = layer(x, inverse=True)
            x = x_new

        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log probability of input.

        Args:
            x: Input tensor (e.g., batch of images or precomputed features)

        Returns:
            log_prob: Tensor of log probabilities for each input sample.
        """
        z, log_det_j = self.forward(x)

        # Standard Gaussian prior
        log_pz = -0.5 * torch.sum(z**2, dim=1) - 0.5 * self.input_dim * np.log(
            2 * np.pi
        )

        return log_pz + log_det_j


def train_normalizing_flow(mode, dataset_name) -> None:
    """Train the normalizing flow model.

    Args:
        mode: Whether to compute features from "images" or use "features" directlyq
        dataset_name: Name of the dataset to use (e.g., "bdd")
    """
    assert mode in {"images", "features"}

    # Convert to tensor and move to device
    logs = Register(Path(__file__).name, NUM_EPOCHS)

    if mode == "images":
        dataset = ImageDataset(dataset_name)
        train_loader, val_loader, _ = dataset.create_image_dataloaders()

    else:
        dataset = FeatureLoader(dataset_name)
        train_loader, val_loader, _ = dataset.create_feature_dataloaders()

    feature_min, feature_max = compute_normalization_stats(train_loader, device)

    hidden_dim = 128
    num_layers = 8
    # Initialize model and move to device
    model = RealNVP(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        mode=mode,
        feature_min=feature_min,
        feature_max=feature_max,
    ).to(device)
    logs.info(model.input_dim)
    logs.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logs.info(f"Model moved to: {device}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    # Training loop

    best_val_loss = float("inf")

    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0

        total_batches = len(train_loader)
        quarter_batches = total_batches // 4
        # Mini-batch training
        for i, (batch,) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}", total=quarter_batches)
        ):
            if i >= quarter_batches:
                break
            batch = batch.to(device)
            optimizer.zero_grad()
            # Compute negative log likelihood
            log_prob = model.log_prob(batch)
            loss = -torch.mean(log_prob)

            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_train_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_train_loss / num_batches
        optimizer.param_groups[0]["lr"]  # Get current learning rate
        # logs.logging(epoch, avg_train_loss, current_lr)

        # Validation
        model.eval()

        with torch.no_grad():
            model.eval()  # Set model to evaluation mode
            val_losses_per_batch = []

            for (batch,) in val_loader:
                batch = batch.to(device)
                val_log_prob = model.log_prob(batch)
                val_loss_batch = -torch.mean(val_log_prob)
                val_losses_per_batch.append(val_loss_batch.item())

        val_loss = sum(val_losses_per_batch) / len(val_losses_per_batch)
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        # Save best model (move to CPU for saving)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.cpu().state_dict(),  # Save on CPU
                    "input_dim": model.input_dim,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "mode": mode,
                    "dataset": dataset_name,
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "epoch": epoch,
                    "device": str(device),
                },
                logs.get_path() + "normalizing_flow_model.pth",
            )
            model.to(device)  # Move back to device

    # print(f"Model saved to: {logs.get_path()}")

    # Final GPU memory info
    torch.cuda.is_available()

    logs.finished(network=model, train_losses=train_losses, val_losses=val_losses)


def main() -> None:
    """Train Normalizing flow on image dataset."""
    train_normalizing_flow(mode="features", dataset_name="bdd")


if __name__ == "__main__":
    main()
