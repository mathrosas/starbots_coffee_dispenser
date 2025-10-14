#!/usr/bin/env python3
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from custom_msgs.msg import InferenceResult, SegmentationResult, Yolov11Inference, Yolov11Segmentation
import cv2
import numpy as np
import os

class YoloObjectDetection(Node):
    def __init__(self) -> None:
        super().__init__('yolo_object_detection')

        # --- Models ---
        # Keep your original detector (optional – we now use the seg model for both boxes+masks)
        self.det_model = YOLO('/home/user/ros2_ws/src/starbot_coffee_dispenser/object_detection/data/yolo11n.pt')

        self.model = YOLO('/home/user/ros2_ws/src/starbot_coffee_dispenser/object_detection/data/yolo11n-seg.pt')

        self.bridge = CvBridge()

        # --- Inference knobs ---
        # Cup only, conf >= 0.04
        self.target_label = 'cup'
        self.min_conf = 0.04
        self.iou = 0.50
        self.imgsz = 512

        # Resolve 'cup' class id from the seg model's label map
        self.cup_id = self._resolve_class_id(self.model.names, self.target_label)
        self.classes_filter = [self.cup_id] if self.cup_id is not None else None
        self.get_logger().info(
            f'cup_id={self.cup_id} min_conf={self.min_conf} classes_filter={self.classes_filter} imgsz={self.imgsz}'
        )

        # --- ROS I/O ---
        self.sub = self.create_subscription(Image, '/camera_depth_sensor/image_raw', self.camera_callback, 10)

        # Detections (boxes)
        self.pub_det_msg  = self.create_publisher(Yolov11Inference, '/Yolov11_Inference', 1)
        self.pub_det_img  = self.create_publisher(Image, '/inference_result', 1)

        # Segmentations (masks)
        self.pub_seg_msg  = self.create_publisher(Yolov11Segmentation, '/Yolov11_segmentation', 1)
        self.pub_seg_img  = self.create_publisher(Image, '/segmentation_result', 1)

    # ---------------- utils ----------------
    def _resolve_class_id(self, names, target: str):
        t = target.lower()
        if isinstance(names, dict):
            for k, v in names.items():
                if str(v).lower() == t:
                    return int(k)
        else:
            for i, v in enumerate(names):
                if str(v).lower() == t:
                    return int(i)
        return None

    def _name_from_id(self, cid: int) -> str:
        names = self.model.names
        if isinstance(names, dict):
            return str(names.get(cid, cid))
        if 0 <= cid < len(names):
            return str(names[cid])
        return str(cid)

    # --------------- callback ---------------
    def camera_callback(self, msg: Image) -> None:
        # ROS -> OpenCV
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge conversion failed: {e}')
            return

        # Small brightness/contrast lift (optional)
        img = cv2.convertScaleAbs(img, alpha=1.35, beta=18)

        # One pass with the segmentation model (returns boxes + masks)
        try:
            results = self.model(
                img,
                conf=self.min_conf,
                iou=self.iou,
                imgsz=self.imgsz,
                classes=self.classes_filter,  # cup only
                device='cpu',
                verbose=False,
            )
        except Exception as e:
            self.get_logger().error(f'YOLO seg inference error: {e}')
            return

        r0 = results[0] if results else None
        if r0 is None:
            self.get_logger().info('no result object')
            return

        boxes = getattr(r0, 'boxes', None)
        masks = getattr(r0, 'masks', None)  # ultralytics Masks object or None

        # ---------------- build messages ----------------
        det_msg = Yolov11Inference()
        det_msg.header = msg.header

        seg_msg = Yolov11Segmentation()
        seg_msg.header = msg.header

        kept = 0
        annotated = r0.plot()  # Ultralytics draws boxes + masks when present

        if boxes is not None and len(boxes) > 0:
            xyxy  = boxes.xyxy.cpu().numpy().astype(np.float32)
            confs = boxes.conf.cpu().numpy().astype(np.float32) if boxes.conf is not None else np.zeros((len(xyxy),), np.float32)
            clses = boxes.cls.cpu().numpy().astype(np.int32)  if boxes.cls  is not None else -np.ones((len(xyxy),), np.int32)

            for idx, ((x1, y1, x2, y2), score, cls_id) in enumerate(zip(xyxy, confs, clses)):
                # Keep ONLY cups with score >= min_conf
                if self._name_from_id(int(cls_id)).lower() != self.target_label or float(score) < self.min_conf:
                    continue
                kept += 1

                # ----- detection record -----
                det = InferenceResult()
                det.class_name = self._name_from_id(int(cls_id))
                det.left, det.top, det.right, det.bottom = int(x1), int(y1), int(x2), int(y2)
                det.box_width  = det.right - det.left
                det.box_height = det.bottom - det.top
                det.x = det.left + det.box_width  / 2.0
                det.y = det.top  + det.box_height / 2.0
                if hasattr(det, 'class_id'): det.class_id = int(cls_id)
                if hasattr(det, 'score'):    det.score    = float(score)
                det_msg.yolov11_inference.append(det)

                # ----- segmentation record (use box center/size like your example) -----
                seg = SegmentationResult()
                seg.class_id   = int(cls_id)
                seg.class_name = det.class_name
                seg.x          = float((x1 + x2) / 2.0)
                seg.y          = float((y1 + y2) / 2.0)
                seg.width      = float(x2 - x1)
                seg.height     = float(y2 - y1)
                seg_msg.yolov11_segmentation.append(seg)

        if kept == 0:
            self.get_logger().info(f'no {self.target_label} >= {self.min_conf:.2f} this frame')
        else:
            self.get_logger().info(f'kept {kept} {self.target_label}(s) with conf ≥ {self.min_conf:.2f}')

        # ---------------- publish ----------------
        # Annotated image (boxes + masks)
        seg_img = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        seg_img.header = msg.header
        self.pub_seg_img.publish(seg_img)  # /segmentation_result
        self.pub_det_img.publish(seg_img)  # also push to /inference_result for convenience

        # Structured messages
        self.pub_det_msg.publish(det_msg)
        self.pub_seg_msg.publish(seg_msg)


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
