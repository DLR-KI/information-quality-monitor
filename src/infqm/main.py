# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Main file for Image Quality Monitor ROS2 Node."""

import importlib
import inspect
import os
import sys
from pathlib import Path

import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import LivelinessPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from infqm.base_metric import BaseMetric
from infqm.normalizing_flow.eval import RealNVPEvaluator

qos_profile = QoSProfile(depth=10)
qos_profile.liveliness = LivelinessPolicy.AUTOMATIC
qos_profile.liveliness_lease_duration.sec = 1  # lease duration 1 second
qos_profile.liveliness_lease_duration.nanosec = 0
qos_profile.reliability = ReliabilityPolicy.RELIABLE


class ImageQualityAnalyzer(Node):
    """Node to measure information quality in incoming image stream."""

    def __init__(self) -> None:
        """Initialize the ImageQualityAnalyzer ROS Node."""
        super().__init__("image_quality_analyzer")

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Get input topic from environment variable
        self.input_topic = os.getenv("INPUT_TOPIC", "/image")

        # Create subscription to image topic
        self.subscription = self.create_subscription(
            Image, self.input_topic, self.image_callback, 10
        )

        # Load all metric classes and create publishers
        self.metrics = []
        self.node_publishers = {}

        self.load_metrics()

        self.get_logger().info(
            f"Image Quality Analyzer started, subscribing to: {self.input_topic}"
        )
        metric_names = [m.get_name() for m in self.metrics]
        self.get_logger().info(f"Loaded {len(self.metrics)} metrics: {metric_names}")

        model_path = (
            Path(__file__).parent.parent.parent
            / "network_logs"
            / f"train_{os.getenv('model_id')}"
        )
        self.normalizing_flow = RealNVPEvaluator(str(model_path))

        # Create publisher for this metric
        self.normalizing_flow_publisher = self.create_publisher(
            Float32, "log_likelihood", 10
        )

        self.preprocessed_image_publisher = self.create_publisher(
            Image, "preprocessed_image", 10
        )

    def load_metrics(self) -> None:
        """Dynamically load all metric classes from the metrics directory."""
        # Get the directory where this script is located
        script_dir = Path(__file__).parent
        metrics_dir = script_dir / "kpi_num"

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
                module = importlib.import_module(module_name)

                # Find all classes in the module that inherit from BaseMetric
                for _, obj in inspect.getmembers(module):
                    if (
                        inspect.isclass(obj)
                        and issubclass(obj, BaseMetric)
                        and obj != BaseMetric
                    ):
                        # Instantiate the metric class
                        metric_instance = obj()
                        self.metrics.append(metric_instance)

                        # Create publisher for this metric
                        self.node_publishers[metric_instance.get_name()] = (
                            self.create_publisher(
                                Float32, metric_instance.get_topic_name(), 10
                            )
                        )

                        self.get_logger().info(
                            f"Loaded metric: {metric_instance.get_name()}"
                        )

            except (ImportError, AttributeError, TypeError) as e:
                self.get_logger().error(
                    f"Error loading metric module {module_name}: {e!s}"
                )

        self.metrics.sort(key=lambda m: m.get_name())

    def safe_imgmsg_to_cv2(self, msg):
        """Safely convert ROS Image message to OpenCV image.

        This method attempts to convert a ROS Image message to an OpenCV
        image, while handling potential issues.

        Args:
            msg: ROS Image message to convert

        Returns:
            OpenCV image if conversion is successful, otherwise None.
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            if cv_image is None or cv_image.size == 0:
                return None

            if cv_image.shape[2] == 4:
                # Remove alpha channel
                cv_image = cv_image[:, :, :3]

            if len(cv_image.shape) != 3 or cv_image.shape[2] != 3:
                return None

            out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            out_msg.header = msg.header
            self.preprocessed_image_publisher.publish(out_msg)

            return cv_image

        except CvBridgeError:
            return None

    def image_callback(self, msg) -> None:
        """Apply assessment list to incoming image stream.

        Args:
            msg: ROS Image message containing the input image.
        """
        try:
            # Convert ROS Image message to OpenCV image
            # print(msg)
            cv_image = self.safe_imgmsg_to_cv2(msg)
            latent = []

            if cv_image is not None:
                # Calculate all metrics and publish them
                for metric in self.metrics:
                    try:
                        value = metric.calculate(cv_image)
                        self.publish_metric(metric.get_name(), value)
                        latent.append(value)
                        # self.get_logger().info(f"[[{metric.get_name()}]]: {value}")

                    except (ImportError, AttributeError, TypeError) as e:
                        self.get_logger().error(
                            f"Error calculating {metric.get_name()}: {e!s}"
                        )

                likelihood = self.normalizing_flow.test_single_point(latent)[1]
                msg = Float32()
                msg.data = float(likelihood)
                self.normalizing_flow_publisher.publish(msg)

        except (ImportError, AttributeError, TypeError) as e:
            print("Skipping Image '%s': %s", msg, e)

    def publish_metric(self, metric_name, value) -> None:
        """Publish a metric value to the specified topic.

        Args:
            metric_name: Name of the metric to publish
            value: Value of the metric to publish.
        """
        if metric_name in self.node_publishers:
            msg = Float32()
            msg.data = float(value)
            self.node_publishers[metric_name].publish(msg)


def main(args=None) -> None:
    """Start Image Quality Analyzer Node.

    Args:
        args: Optional command-line arguments to pass to rclpy.init()
    """
    rclpy.init(args=args)

    try:
        image_quality_analyzer = ImageQualityAnalyzer()
        rclpy.spin(image_quality_analyzer)
    except KeyboardInterrupt:
        pass
    finally:
        if "image_quality_analyzer" in locals():
            image_quality_analyzer.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
