# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Description."""

import csv
import logging
import os
import pathlib
import pickle
import shutil
import sys
from datetime import datetime, timezone

import matplotlib.pyplot as plt


class Register:
    """Class to log the training progress.

    The file stores the training process over time as well as a copy of
    the current python script
    """

    def __init__(self, file, training_iter) -> None:
        """Logger initialization.

        Args:
            file: Name of the current python script (e.g., "train.py")
            training_iter: Total number of training iterations (epochs).
        """
        self.training_iter = training_iter

        file_dir = pathlib.Path.cwd()

        file_name__ = os.path.splitext(file)[0]  # Get the string 'cnn.py'

        # id for this particular experiment
        mid = f"{file_name__}_{datetime.now(tz=timezone.utc):%Y%m%d_%H%M%S}"

        # Make directory according to ID
        self.path = file_dir / f"network_logs/{mid}/"
        pathlib.Path(self.path).mkdir(parents=True, exist_ok=True)

        # save a copy of this script
        shutil.copy(
            os.path.join(pathlib.Path(os.path.realpath(__file__)).parent, "train.py"),
            self.path / "_script.py",
        )
        shutil.copy(
            os.path.join(pathlib.Path(os.path.realpath(__file__)).parent, "logger.py"),
            self.path / "logger.py",
        )

        # region Logging
        self.logger = logging.getLogger("new_logger")
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        fh = logging.FileHandler(self.path / "_output.log")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.logger.info("ModelID: %s", mid)
        self.start_time = datetime.now(tz=timezone.utc)

    def logging(self, i, total_loss, stepsize) -> None:
        """Log the current training progress.

        Args:
            i:              Current training epoch
            total_loss:     Current training loss
            stepsize:       Current learning rate
        """
        self.logger.info(
            "Iter %d: Progress %.2f%% Loss = %s, stepsize = %s, Time = %s",
            i,
            i / self.training_iter * 100,
            total_loss,
            stepsize,
            datetime.now(tz=timezone.utc) - self.start_time,
        )

    def info(self, text) -> None:
        """Directly add something to the logger.

        Args:
            text: Text to log.
        """
        self.logger.info(text)

    def save_preprocessor(self, scaler) -> None:
        """Save the scaler for later use.

        Args:
            scaler: Scaler object to save (e.g., fitted MinMaxScaler).
        """
        with pathlib.Path(self.path / "feature_scaler.pkl").open("wb") as f:
            pickle.dump(scaler, f)

    def finished(self, network, train_losses, val_losses) -> None:
        """Called after training is finished.

        Logs the total number of current network parameter, saves copies
        of the python script and also creates a log-log plot of the
        training loss over time.

        Args:
            network: The trained normalizing flow model
            train_losses: List of training losses over epochs
            val_losses: List of validation losses over epochs
        """
        pytorch_total_params = sum(p.numel() for p in network.parameters())

        # torch.save(network.state_dict(), self.path / "Normalizing_Flow.pth")
        self.logger.info(
            "Finished! Number of Model parameters: %s", pytorch_total_params
        )
        plt.rcParams["figure.figsize"] = (17, 4)
        # Plot training curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Negative Log Likelihood")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.path / "training_progress.png", dpi=150, bbox_inches="tight")

    def get_path(self):
        """Return Path of Logging folder.

        Returns:
            str: Path to the logging folder.
        """
        return self.path

    def copy_training_data(self, cvs_file_path):
        """Copy an excpert of the training data into logs folder.

        Args:
            cvs_file_path: Path to the CSV file containing the training data.

        Returns:
            str: Path to the copied CSV file containing the training data excerpt.
        """
        dest_path = os.path.join(
            self.path, pathlib.Path(cvs_file_path.split("/")[-1]).name
        )
        with (
            pathlib.Path(cvs_file_path).open(newline="", encoding="utf-8") as infile,
            pathlib.Path(dest_path).open("w", newline="", encoding="utf-8") as outfile,
        ):
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            for i, row in enumerate(reader):
                writer.writerow(row)
                if i >= 99:  # stop after 100 rows (including header if present)
                    break

        return dest_path
