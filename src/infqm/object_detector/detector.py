# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Object Detector Node for ROS2."""

import os

import rclpy
import torch as th
from conversion import ArcTanConversion
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32


class ObjectDetectorRosNode(Node):
    """Base Class for Object Detector Nodes."""

    def __init__(self) -> None:
        """Initialize all Ros Nodes."""
        super().__init__("detector_node")
        self.declare_parameter("image_topic", os.getenv("INPUT_TOPIC", "/image"))
        self.declare_parameter("confidence_topic", "/log_likelihood")
        self.declare_parameter("output_topic", "/yolo/image_with_boxes")

        self.image_topic = (
            self.get_parameter("image_topic").get_parameter_value().string_value
        )
        self.conf_topic = (
            self.get_parameter("confidence_topic").get_parameter_value().string_value
        )
        self.pub_topic = (
            self.get_parameter("output_topic").get_parameter_value().string_value
        )

        self.bridge = CvBridge()
        self.confidence = 0.5

        self.conversion = ArcTanConversion()

        self.create_subscription(Float32, self.conf_topic, self.conf_callback, 10)

        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

        self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.pub = self.create_publisher(Image, self.pub_topic, 1)

        self.confidence_publisher = self.create_publisher(Float32, "confidence", 10)

    def safe_imgmsg_to_cv2(self, msg):
        """ImgMsg to CV2 conversion with error handling.

        Safely convert ROS Image message to OpenCV image, handling
        potential errors and edge cases.

        Args:
            msg: ROS Image message to convert

        Returns:
            np.ndarray | None: OpenCV image array if successful, None otherwise.
        """
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if cv_image.shape[2] == 4:
            # Remove alpha channel
            cv_image = cv_image[:, :, :3]

        if cv_image is None or cv_image.size == 0:
            return None

        if len(cv_image.shape) != 3 or cv_image.shape[2] != 3:
            return None

        return cv_image

    def conf_callback(self, msg) -> None:
        """Set Confidence threshold and publish it via ROS.

        Args:
            msg: ROS Float32 message containing the new confidence threshold.
        """
        self.confidence = self.conversion(msg.data)

        new_msg = Float32()
        new_msg.data = float(self.confidence)
        self.confidence_publisher.publish(new_msg)


def run_node(detection_node) -> None:
    """Main function to run for a detector.

    Args:
        detection_node: Class of the detector node to run.
    """
    rclpy.init()
    node = detection_node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
