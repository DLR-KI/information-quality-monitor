# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Base class for image quality metrics."""


class BaseMetric:
    """Base class for all image quality metrics.

    All metric classes should inherit from this class.
    """

    def __init__(self) -> None:
        self.name = self.__class__.__name__.lower().replace("metric", "")
        self.topic_name = f"/image_quality/{self.name}"

    def calculate(self, cv_image):
        """Calculate the metric value for the given OpenCV image.

        Args:
            cv_image: OpenCV image (numpy array)

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement calculate() method")

    def get_name(self):
        """Return the metric name.

        Returns:
            str: The name of the metric.
        """
        return self.name

    def get_topic_name(self):
        """Return the ROS topic name for this metric.

        Returns:
            str: The ROS topic name for this metric.
        """
        return self.topic_name

    def get_output_dim(self) -> int:
        """Returns output dimension of the metric, defaults to scalars.

        Returns:
            int: The output dimension of the metric (default is 1 for scalar metrics).
        """
        return 1

    def tensor_rgb2gray(self, image_batch):
        """Transforms PyTorch RGB images into Grayscale values.

        Args:
            image_batch: PyTorch tensor

        Returns:
            Grayscale version of the input tensor.

        Raises:
            ValueError: If the input tensor does not have the expected shape.
        """
        if image_batch.dim() == 4 and image_batch.shape[1] == 3:
            # Assuming RGB format with shape (3, H, W)
            # Using standard luminance conversion weights
            gray = (
                0.299 * image_batch[:, 0]
                + 0.587 * image_batch[:, 1]
                + 0.114 * image_batch[:, 2]
            )
        elif image_batch.dim() == 4 and image_batch.shape[1] == 1:
            # Grayscale batch with shape (batch_size, 1, H, W)
            gray = image_batch[:, 0]
        else:
            raise ValueError(
                f"Expected 4D tensor with shape (B, C, H, W), \
                got shape {image_batch.shape}"
            )

        return gray
