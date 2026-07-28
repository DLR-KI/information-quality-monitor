# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Violin plots for OOD detection.

Script to create violin plots for OOD detection results across different
datasets.
"""

import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection

plt.style.use("utils/figurestil.mplstyle")


def plot_ood_violins(model) -> None:
    """Handle the creation of violin plots for OOD detection results.

    Create a modern violin plot for OOD data across different datasets
    for a given model.

    Args:
        model: Name of the model directory to load

    Raises:
        ValueError: If no files are found matching the pattern
    """
    folder_path = "ood_data/results/"  # Directory containing the CSV files

    # Find all CSV files matching the pattern for the given model
    files = glob.glob(os.path.join(folder_path, f"OOD_{model}_*.csv"))

    if not files:
        raise ValueError("No files found matching pattern")

    # Load data from all matching files
    data_list = []

    dataset_rename = {
        "BDD100k": "BDD100k",
        "Zenseact Open Dataset (ZOD)": "ZOD",
        "carla": "CARLA",
    }

    for file_path in files:
        # Extract dataset name from filename
        filename = Path(file_path).stem  # Get filename without extension
        # Parse: OOD_{model}_{dataset}
        parts = filename.split("_")
        dataset = "_".join(parts[2:]).split("_")[1]  # Everything after OOD_{model}_

        dataset = dataset_rename[dataset]

        # Load CSV
        df = pd.read_csv(file_path)
        df["Dataset"] = dataset
        data_list.append(df)

    def keep_95_percent(df, column):
        """Keep only data between 2.5th and 97.5th percentiles.

        Args:
            df: DataFrame containing the data
            column: Column name to calculate percentiles on

        Returns:
            dataframe: Filtered DataFrame with only the middle 95% of data
        """
        lower = df[column].quantile(0.025)
        upper = df[column].quantile(0.975)
        return df[(df[column] >= lower) & (df[column] <= upper)]

    # Apply per dataset

    # Combine all data
    combined_data = pd.concat(data_list, ignore_index=True)

    combined_data_95 = (
        combined_data
        .groupby("Dataset")
        .apply(lambda x: keep_95_percent(x, "log_likelihood"))
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots()

    sns.violinplot(
        data=combined_data_95,
        x="Dataset",
        y="log_likelihood",
        hue="Dataset",  # same as x, to apply the palette correctly
        palette=["#385823", "#e2edbf", "#e2edbf"],
        ax=ax,
        inner="box",
        saturation=0.8,
        cut=0,
        bw_adjust=0.5,
        legend=False,  # hide redundant legend
    )

    for dataset in combined_data_95["Dataset"].unique():
        combined_data_95[combined_data_95["Dataset"] == dataset][
            "log_likelihood"
        ].median()
    # Manually adjust inner box colors for consistency

    for _, artist in enumerate(ax.findobj(LineCollection)):
        artist.set_color("black")  # or any color you prefer
        artist.set_linewidth(1.0)

    # Styling for modern look
    # ax.set_xlabel("Dataset", fontsize=13, fontweight="bold")
    ax.set_ylabel("Log-Likelihood", fontweight="bold")

    # ax.set_yscale("symlog", linthresh=10)
    ax.set_ylim(-20, 50)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y")

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Remove top and right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(folder_path + "ood_violins.jpg", bbox_inches="tight", dpi=200)


def main():
    """Main function to execute the plotting of OOD detection results."""
    # Plot for a specific model
    plot_ood_violins("20251013_090123")

    # You can also save the figure
    # fig.savefig('ood_violins.png', dpi=300, bbox_inches='tight')


# Example usage:
if __name__ == "__main__":
    main()
