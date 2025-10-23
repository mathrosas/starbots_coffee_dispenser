#!/usr/bin/env python3
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from cv_bridge import CvBridge
from custom_msgs.msg import InferenceResult, SegmentationResult, Yolov11Inference, Yolov11Segmentation
from tf2_ros import TransformBroadcaster
import cv2
import numpy as np

class YoloObjectDetection(Node):
    def __init__(self) -> None:
        super().__init__('yolo_object_detection')

        # --- Models ---
        self.det_model = YOLO('/home/user/ros2_ws/src/starbot_coffee_dispenser/object_detection/data/yolo11n.pt')
        self.model = YOLO('/home/user/ros2_ws/src/starbot_coffee_dispenser/object_detection/data/yolo11n-seg.pt')

        self.bridge = CvBridge()

        # --- Inference knobs ---
        self.target_label = 'cup'
        self.min_conf = 0.04
        self.iou = 0.50
        self.imgsz = 512

        # Resolve 'cup' class id
        self.cup_id = self._resolve_class_id(self.model.names, self.target_label)
        self.classes_filter = [self.cup_id] if self.cup_id is not None else None
        self.get_logger().info(
            f'cup_id={self.cup_id} min_conf={self.min_conf} classes_filter={self.classes_filter} imgsz={self.imgsz}'
        )

        # --- Camera intrinsics & depth cache ---
        self.fx = self.fy = self.cx = self.cy = None
        self.depth_img = None
        self.depth_frame_id = None

        # --- TF broadcaster ---
        self.cup_frame = 'cup'
        self.tf_broadcaster = TransformBroadcaster(self)

        # --- ROS I/O ---
        self.sub       = self.create_subscription(Image, '/wrist_rgbd_depth_sensor/image_raw', self.camera_callback, 10)
        self.sub_depth = self.create_subscription(Image, '/wrist_rgbd_depth_sensor/depth/image_raw', self.depth_callback, 10)
        self.sub_info  = self.create_subscription(CameraInfo, '/wrist_rgbd_depth_sensor/camera_info', self.info_callback, 10)

        self.pub_det_msg = self.create_publisher(Yolov11Inference, '/Yolov11_Inference', 1)
        self.pub_det_img = self.create_publisher(Image, '/inference_result', 1)

        self.pub_seg_msg = self.create_publisher(Yolov11Segmentation, '/Yolov11_segmentation', 1)
        self.pub_seg_img = self.create_publisher(Image, '/segmentation_result', 1)

        self.pub_pose    = self.create_publisher(PoseStamped, '/cup_pose', 1)

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

    # --------------- info/depth callbacks ---------------
    def info_callback(self, msg: CameraInfo) -> None:
        K = msg.k  # [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        self.fx, self.fy, self.cx, self.cy = float(K[0]), float(K[4]), float(K[2]), float(K[5])

    def depth_callback(self, msg: Image) -> None:
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Depth CvBridge conversion failed: {e}')
            return
        self.depth_img = depth
        self.depth_frame_id = msg.header.frame_id

    # --------------- main RGB callback ---------------
    def camera_callback(self, msg: Image) -> None:
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge conversion failed: {e}')
            return

        img = cv2.convertScaleAbs(img, alpha=1.35, beta=18)

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
        masks = getattr(r0, 'masks', None)

        # ---------------- build messages ----------------
        det_msg = Yolov11Inference()
        det_msg.header = msg.header

        seg_msg = Yolov11Segmentation()
        seg_msg.header = msg.header

        kept = 0
        kept_idxs = []
        annotated = r0.plot()  # Ultralytics draws ONLY boxes/masks (no pose overlay here)

        xyxy = confs = clses = None
        if boxes is not None and len(boxes) > 0:
            xyxy  = boxes.xyxy.cpu().numpy().astype(np.float32)
            confs = boxes.conf.cpu().numpy().astype(np.float32) if boxes.conf is not None else np.zeros((len(xyxy),), np.float32)
            clses = boxes.cls.cpu().numpy().astype(np.int32)  if boxes.cls  is not None else -np.ones((len(xyxy),), np.int32)

            for idx, ((x1, y1, x2, y2), score, cls_id) in enumerate(zip(xyxy, confs, clses)):
                if self._name_from_id(int(cls_id)).lower() != self.target_label or float(score) < self.min_conf:
                    continue
                kept += 1
                kept_idxs.append(idx)

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

                seg = SegmentationResult()
                seg.class_id   = int(cls_id)
                seg.class_name = det.class_name
                seg.x          = float((x1 + x2) / 2.0)
                seg.y          = float((y1 + y2) / 2.0)
                seg.width      = float(x2 - x1)
                seg.height     = float(y2 - y1)
                seg_msg.yolov11_segmentation.append(seg)

        # ---------------- pose estimation (best cup) ----------------
        pose = None
        if kept > 0 and (self.depth_img is not None) and (self.fx is not None):
            best_idx = max(kept_idxs, key=lambda i: float(confs[i]) if confs is not None else 0.0)

            H_rgb, W_rgb = img.shape[:2]
            if masks is not None and hasattr(masks, "data") and masks.data is not None and masks.data.shape[0] > best_idx:
                m = masks.data[best_idx].cpu().numpy()
                mask = (cv2.resize(m, (W_rgb, H_rgb), interpolation=cv2.INTER_NEAREST) > 0.5)
            else:
                x1, y1, x2, y2 = xyxy[best_idx]
                mask = np.zeros((H_rgb, W_rgb), dtype=bool)
                mask[int(max(0, y1)):int(min(H_rgb, y2)), int(max(0, x1)):int(min(W_rgb, x2))] = True

            depth = self.depth_img
            H_d, W_d = depth.shape[:2]
            if (H_d, W_d) != (H_rgb, W_rgb):
                mask = cv2.resize(mask.astype(np.uint8), (W_d, H_d), interpolation=cv2.INTER_NEAREST).astype(bool)

            d = depth[mask]
            if d.size > 0:
                if d.dtype == np.uint16 or d.dtype == np.uint32:
                    Z_vals = d.astype(np.float32) * 0.001
                else:
                    Z_vals = d.astype(np.float32)

                Z_vals = Z_vals[np.isfinite(Z_vals) & (Z_vals > 0.05) & (Z_vals < 10.0)]
                if Z_vals.size > 0:
                    Z = float(np.median(Z_vals))
                    ys, xs = np.nonzero(mask)
                    if xs.size > 0:
                        u = float(np.median(xs))
                        v = float(np.median(ys))
                        X = (u - self.cx) * Z / self.fx
                        Y = (v - self.cy) * Z / self.fy

                        pose = PoseStamped()
                        pose.header.frame_id = self.depth_frame_id if self.depth_frame_id else msg.header.frame_id
                        pose.header.stamp = msg.header.stamp
                        pose.pose.position.x = X
                        pose.pose.position.y = Y
                        pose.pose.position.z = Z
                        pose.pose.orientation.x = 0.0
                        pose.pose.orientation.y = 0.0
                        pose.pose.orientation.z = 0.0
                        pose.pose.orientation.w = 1.0

        # If we have a pose, publish it and broadcast TF (no overlay drawn on image)
        if pose is not None:
            self.pub_pose.publish(pose)
            self.get_logger().info(
                f'cup pose -> frame={pose.header.frame_id} pos=({pose.pose.position.x:.3f}, {pose.pose.position.y:.3f}, {pose.pose.position.z:.3f})'
            )
            t = TransformStamped()
            t.header.stamp = pose.header.stamp
            t.header.frame_id = pose.header.frame_id
            t.child_frame_id = self.cup_frame
            t.transform.translation.x = pose.pose.position.x
            t.transform.translation.y = pose.pose.position.y
            t.transform.translation.z = pose.pose.position.z
            t.transform.rotation = pose.pose.orientation
            self.tf_broadcaster.sendTransform(t)
        else:
            if kept == 0:
                self.get_logger().info(f'no {self.target_label} >= {self.min_conf:.2f} this frame')
            elif (self.depth_img is None) or (self.fx is None):
                self.get_logger().warn('Depth and/or CameraInfo not ready; skipping pose publish.')
            else:
                self.get_logger().warn('Pose could not be computed for this frame.')

        # ---------------- publish images & structured messages ----------------
        seg_img = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        seg_img.header = msg.header
        self.pub_seg_img.publish(seg_img)   # /segmentation_result
        self.pub_det_img.publish(seg_img)   # /inference_result
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
