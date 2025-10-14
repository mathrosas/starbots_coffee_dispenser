#!/usr/bin/env python3
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from custom_msgs.msg import InferenceResult, Yolov11Inference
import cv2
import numpy as np

class YoloObjectDetection(Node):
    def __init__(self) -> None:
        super().__init__('yolo_object_detection')

        # Model
        self.model = YOLO('/home/user/ros2_ws/src/starbot_coffee_dispenser/object_detection/data/yolo11n.pt')
        self.bridge = CvBridge()

        # We only want cups with conf >= 0.04
        self.target_label = 'cup'
        self.min_conf = 0.02
        self.iou = 0.50
        self.imgsz = 512

        # Resolve class id for "cup" and filter the model to that class
        self.cup_id = None
        names = self.model.names
        if isinstance(names, dict):
            for k, v in names.items():
                if str(v).lower() == self.target_label:
                    self.cup_id = int(k); break
        else:
            for i, n in enumerate(names):
                if str(n).lower() == self.target_label:
                    self.cup_id = int(i); break

        self.classes_filter = [self.cup_id] if self.cup_id is not None else None
        self.get_logger().info(
            f'cup_id={self.cup_id} min_conf={self.min_conf} classes_filter={self.classes_filter} imgsz={self.imgsz}'
        )

        # ROS I/O
        self.sub = self.create_subscription(Image, '/camera_depth_sensor/image_raw', self.camera_callback, 10)
        self.pub_det = self.create_publisher(Yolov11Inference, '/Yolov11_Inference', 1)
        self.pub_img = self.create_publisher(Image, '/inference_result', 1)

    def _name_from_id(self, cid: int) -> str:
        names = self.model.names
        if isinstance(names, dict):
            return str(names.get(cid, cid))
        if 0 <= cid < len(names):
            return str(names[cid])
        return str(cid)

    def camera_callback(self, msg: Image) -> None:
        # Convert ROS -> OpenCV
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge conversion failed: {e}')
            return

        # Mild enhancement (optional)
        img = cv2.convertScaleAbs(img, alpha=1.35, beta=18)

        # YOLO inference (already class-filtered to cup, min conf = 0.04)
        try:
            results = self.model(
                img,
                conf=self.min_conf,
                iou=self.iou,
                imgsz=self.imgsz,
                classes=self.classes_filter,
                device='cpu',
                verbose=False,
            )
        except Exception as e:
            self.get_logger().error(f'YOLO inference error: {e}')
            return

        det_msg = Yolov11Inference()
        det_msg.header = msg.header

        r0 = results[0] if results else None
        boxes = getattr(r0, 'boxes', None)

        annotated = img.copy()
        kept = 0

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
            confs = boxes.conf.cpu().numpy().astype(np.float32) if boxes.conf is not None else np.zeros((len(xyxy),), np.float32)
            clses = boxes.cls.cpu().numpy().astype(np.int32) if boxes.cls is not None else -np.ones((len(xyxy),), np.int32)

            for (x1, y1, x2, y2), score, cls_id in zip(xyxy, confs, clses):
                # Keep ONLY cups with score >= 0.04
                if self._name_from_id(int(cls_id)).lower() == self.target_label and float(score) >= self.min_conf:
                    kept += 1

                    inf = InferenceResult()
                    inf.class_name = self._name_from_id(int(cls_id))
                    inf.left, inf.top, inf.right, inf.bottom = int(x1), int(y1), int(x2), int(y2)
                    inf.box_width = inf.right - inf.left
                    inf.box_height = inf.bottom - inf.top
                    inf.x = inf.left + inf.box_width / 2.0
                    inf.y = inf.top + inf.box_height / 2.0
                    if hasattr(inf, 'class_id'):
                        inf.class_id = int(cls_id)
                    if hasattr(inf, 'score'):
                        inf.score = float(score)
                    det_msg.yolov11_inference.append(inf)

                    cv2.rectangle(annotated, (inf.left, inf.top), (inf.right, inf.bottom), (255, 255, 0), 2)
                    cv2.putText(annotated, f'cup {score:.2f}', (inf.left, max(0, inf.top - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

        if kept == 0:
            self.get_logger().info('no cup >= 0.04 this frame')
        else:
            self.get_logger().info(f'kept {kept} cup detection(s) with conf >= {self.min_conf}')

        # Publish image + detections
        out = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        out.header = msg.header
        self.pub_img.publish(out)
        self.pub_det.publish(det_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YoloObjectDetection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
