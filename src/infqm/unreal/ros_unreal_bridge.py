# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""ROS Unreal Bridge Module.

ROS Node to receive camera images from Unreal Engine and publish them as
ROS messages.
"""

import os
import socket
import struct

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

HOST = "0.0.0.0"
PORT = 9870


class CameraReceiver(Node):
    """Camera Receiver ROS Node.

    ROS Node to receive camera images from Unreal Engine and publish
    them as ROS messages.
    """

    def __init__(self) -> None:
        """Initialize the CameraNodeReceiver.

        Initialize ROS node and set up socket server to receive images
        from Unreal.
        """
        super().__init__("unreal_camera")
        input_topic = os.getenv("INPUT_TOPIC", "/image")
        self.pub = self.create_publisher(Image, input_topic, 10)
        self.get_logger().info(f"Listening on port {PORT}...")

        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((HOST, PORT))
        self.srv.listen(1)
        self.conn, addr = self.srv.accept()
        self.get_logger().info(f"Unreal connected: {addr}")

        self.create_timer(0.001, self.recv_frame)  # poll fast
        self._buf = b""

    def recv_exact(self, n: int) -> bytes:
        """Receive exactly n bytes from the socket.

        Args:
            n: Number of bytes to receive

        Returns:
            bytes: Received bytes of length n.

        Raises:
            ConnectionResetError: If the connection is lost while receiving.
        """
        data = b""
        while len(data) < n:
            chunk = self.conn.recv(n - len(data))
            if not chunk:
                raise ConnectionResetError("Unreal disconnected")
            data += chunk
        return data

    def recv_frame(self) -> None:
        """Frame Receiver.

        Receive a single frame from Unreal, convert to ROS Image
        message, and publish.
        """
        try:
            header = self.recv_exact(4)
            length = struct.unpack(">I", header)[0]
            jpeg = self.recv_exact(length)

            np_arr = np.frombuffer(jpeg, np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

            msg = Image()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "unreal_camera"
            msg.height = cv_img_rgb.shape[0]
            msg.width = cv_img_rgb.shape[1]
            msg.encoding = "rgb8"
            msg.is_bigendian = False
            msg.step = cv_img_rgb.shape[1] * 3
            msg.data = cv_img_rgb.tobytes()
            self.pub.publish(msg)
        except (ConnectionResetError, OSError):
            self.get_logger().warn("Connection lost, waiting for reconnect...")
            self.conn, _ = self.srv.accept()


def main() -> None:
    """Start Unreal Camera Receiver Node."""
    rclpy.init()
    node = CameraReceiver()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
