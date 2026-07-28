# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Plot training and validation losses from JSON file."""

import json
import pathlib

import matplotlib.pyplot as plt

plt.style.use("utils/figurestil.mplstyle")


def main() -> None:
    """Main function to load losses from JSON and plot them."""
    base_path = "network_logs/train_20251029_163853/"
    # Load from JSON
    with pathlib.Path(base_path + "losses.json").open("r", encoding="utf-8") as f:
        data = json.load(f)

    train_losses = data["train_losses"]
    val_losses = data["val_losses"]

    plt.plot(val_losses, label="Validation Loss", color="#385723")  # dark green
    plt.plot(train_losses, label="Train Loss", color="#80bb88")  # bright green

    plt.legend()
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log-Likelihood")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.savefig(base_path + "new_training_losses.pdf")


if __name__ == "__main__":
    main()
