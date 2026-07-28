# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Script to convert CSV files with features into JSON format for merging."""

import csv
import json
import os
import pathlib


def get_merge_ready(input_file, dataset_name) -> None:
    """Convert a CSV file with features into JSON format for merging.

    Args:
        input_file: Filename of the input CSV file containing features.
        dataset_name: Name of the dataset.
    """
    dataset = dataset_name
    input_file = "src/normalizing_flow/features/" + input_file

    # Read the CSV data
    with pathlib.Path(input_file).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

    # Get all numeric columns (exclude filename)
    columns = [col for col in reader.fieldnames if col != "filename"]

    # Output directory (optional)
    output_dir = "src/normalizing_flow/features/merger"
    pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)

    # Write one JSON file per column
    for col in columns:
        items = {}
        for row in list(reader):
            filename = row["filename"].split("_")[0]
            value = float(row[col]) if row[col] else None
            items[filename] = {"item_value": value}

        data = {"name": col, "type": "num", "items": items}

        output_path = os.path.join(output_dir, f"dlr_{dataset}_{col}.json")
        with pathlib.Path(output_path).open("w", encoding="utf-8") as out_f:
            json.dump(data, out_f, indent=4)


if __name__ == "__main__":
    get_merge_ready("bdd_test.csv", "bdd_test")
    get_merge_ready("bdd_val.csv", "bdd_val")
    get_merge_ready("bdd_train.csv", "bdd_train")
    get_merge_ready("zod_train.csv", "zod")
