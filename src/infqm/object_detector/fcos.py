# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Fcos Object Detector Node for ROS2."""

import cv2
import torch as th
from detector import ObjectDetectorRosNode, run_node
from torchvision import transforms
from torchvision.models.detection import FCOS_ResNet50_FPN_Weights, fcos_resnet50_fpn


class FCOSRosNode(ObjectDetectorRosNode):
    """Run FCOS Object Detection."""

    def __init__(self) -> None:
        """Load FCOS."""
        super().__init__()
        # Custom Model
        # self.model_path = "my_path.pth"
        # self.model = fcos_resnet50_fpn(
        #     weights=None, weights_backbone=None, num_classes=2
        # ).eval()
        # self.model.load_state_dict(th.load(self.model_path, map_location=self.device))

        self.model = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.DEFAULT).eval()
        self.model.to(self.device)

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        print(self.device)

    def image_callback(self, msg) -> None:
        """Draw bounding box prediction onto image.

        Args:
            msg: ROS Image message containing the input image.
        """
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Convert BGR to RGB for torchvision
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        # Convert to tensor and add batch dimension
        img_tensor = self.preprocess(rgb_img).unsqueeze(0).to(self.device)

        # Run inference
        with th.no_grad():
            pred = self.model(img_tensor)[0]

        # Extract boxes, scores, and labels
        boxes = pred["boxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()

        # Filter by confidence threshold
        valid_indices = scores >= self.confidence
        boxes = boxes[valid_indices]
        scores = scores[valid_indices]
        labels = labels[valid_indices]

        # Draw bounding boxes and labels
        for box, score, label in zip(boxes, scores, labels, strict=False):
            if self.confidence <= score:
                x1, y1, x2, y2 = map(int, box)

                # Draw bounding box
                cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f"{int(label)} {score:.2f}"
                cv2.putText(
                    cv_img,
                    label_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        # Convert back to ROS message and publish
        out_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
        self.pub.publish(out_msg)


if __name__ == "__main__":
    run_node(FCOSRosNode)
