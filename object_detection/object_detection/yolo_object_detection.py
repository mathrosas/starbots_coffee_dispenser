#!/usr/bin/env python3
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
)
from rclpy.duration import Duration
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from custom_msgs.msg import InferenceResult, Yolov11Inference
import cv2
import numpy as np


class YoloCupVaseDetector(Node):
    def __init__(self) -> None:
        super().__init__('yolo_cup_vase_detector')

        # ---------- Fixed configuration (no launch params) ----------
        self.camera_topic = '/camera_depth_sensor/image_raw'
        self.model_path = '/home/user/ros2_ws/src/starbot_coffee_dispenser/object_detection/data/yolo11n.pt'
        self.conf = 0.25   # a bit lower helps in sim
        self.iou = 0.50
        self.target_classes = ['cup', 'vase']

        # ---------- Load YOLO model ----------
        self.model = YOLO(self.model_path)
        self.class_names = self.model.names  # dict or list depending on model

        # Resolve target class IDs from names
        self.target_class_ids = []
        names_iter = (
            self.class_names.items()
            if isinstance(self.class_names, dict)
            else enumerate(self.class_names)
        )
        for k, v in names_iter:
            if v in self.target_classes:
                self.target_class_ids.append(int(k))
        if not self.target_class_ids:
            self.get_logger().warn(
                f"Could not find any of {self.target_classes} in model names; will detect ALL classes."
            )
            self.target_class_ids = None  # means "no filter" to Ultralytics

        # ---------- ROS setup ----------
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, self.camera_topic, self.callback, qos)
        self.pub_img = self.create_publisher(Image, '/barista_1/cup_image', 10)
        self.pub_det = self.create_publisher(Yolov11Inference, '/barista_1/cup_detections', 10)

        # simple manual log throttle (since rclpy lacks throttle_* helpers)
        self._last_log = self.get_clock().now()
        self._log_period = Duration(seconds=2.0)

        classes_str = ",".join(self.target_classes)
        self.get_logger().info(
            f"YOLO model ready — detecting [{classes_str}] @ conf≥{self.conf}, iou={self.iou}"
        )

    def _label_from_id(self, cls_id: int) -> str:
        if isinstance(self.class_names, dict):
            return self.class_names.get(cls_id, str(cls_id))
        if 0 <= cls_id < len(self.class_names):
            return self.class_names[cls_id]
        return str(cls_id)

    def callback(self, msg: Image) -> None:
        # Convert image
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CvBridge failed: {e}")
            return

        # Run YOLO inference
        try:
            results = self.model.predict(
                frame,
                conf=self.conf,
                iou=self.iou,
                classes=self.target_class_ids,
                verbose=False
            )
        except Exception as e:
            self.get_logger().error(f"YOLO inference error: {e}")
            return

        det_msg = Yolov11Inference()
        det_msg.header = msg.header

        r0 = results[0] if results else None
        boxes = getattr(r0, 'boxes', None)
        count = int(len(boxes)) if boxes is not None else 0

        # Collect results
        if boxes is not None and count > 0:
            for box in boxes:
                # xyxy tensor -> numpy
                b = box.xyxy[0].to('cpu').detach().numpy().astype(np.float32)
                cls_id = int(box.cls.item()) if box.cls is not None else -1
                score = float(box.conf.item()) if box.conf is not None else 0.0
                label = self._label_from_id(cls_id)

                inf = InferenceResult()
                if hasattr(inf, 'class_id'):
                    inf.class_id = cls_id
                if hasattr(inf, 'score'):
                    inf.score = score
                if hasattr(inf, 'class_name'):
                    inf.class_name = label

                x1, y1, x2, y2 = map(int, b[:4])
                inf.left, inf.top, inf.right, inf.bottom = x1, y1, x2, y2
                inf.box_width = x2 - x1
                inf.box_height = y2 - y1
                inf.x = x1 + inf.box_width / 2.0
                inf.y = y1 + inf.box_height / 2.0

                det_msg.yolov11_inference.append(inf)

        # Annotate and publish image
        annotated = r0.plot() if r0 is not None else frame
        cv2.putText(
            annotated, f'dets={count}', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA
        )
        out = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        out.header = msg.header
        self.pub_img.publish(out)

        # Publish detections (even if empty)
        self.pub_det.publish(det_msg)

        # Throttled info log (rclpy DIY)
        now = self.get_clock().now()
        if (now - self._last_log) > self._log_period:
            self.get_logger().info(f"Detected {count} object(s)")
            self._last_log = now


def main(args=None):
    rclpy.init(args=args)
    node = YoloCupVaseDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
