#!/usr/bin/env python3

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
import tf2_geometry_msgs
import tf2_ros
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PointStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray

from custom_msgs.msg import DetectedObjects
from object_detection.geometry import estimate_depth_for_circle, project_pixel_to_3d
from object_detection.perception_core import Candidate2D, CupholderPerception, PerceptionConfig
from object_detection.tracker import Detection3D, StableTracker

# Fixed tray-plane normalization settings (non-configurable on purpose).
_PLANE_ROI_X = (-0.70, -0.20)
_PLANE_ROI_Y = (-0.20, 0.50)
_PLANE_ROI_Z = (-0.70, -0.20)
_PLANE_MAX_POINTS = 5000
_PLANE_RANSAC_ITERS = 80
_PLANE_DIST_THRESH_M = 0.008
_PLANE_MIN_INLIERS = 250
_PLANE_MIN_INLIER_RATIO = 0.10
_PLANE_MIN_NZ = 0.75
_PLANE_MAX_AGE_SEC = 1.5
_PLANE_MAX_Z_CORRECTION_M = 0.03
_PLANE_Z_OFFSET_M = 0.0

# Occupancy check: detections with enough 3D points above the cupholder area
# are treated as occupied and filtered out.
_OCCUPANCY_XY_MARGIN_M = 0.01
_OCCUPANCY_Z_ABOVE_M = 0.04
_OCCUPANCY_MIN_POINTS_ABOVE = 20
_OCCUPANCY_MAX_CLOUD_AGE_SEC = 0.75


class ObjectDetectionNode(Node):
    def __init__(self) -> None:
        super().__init__("object_detection")

        # Inputs
        self.declare_parameter("color_topic", "/wrist_rgbd_depth_sensor/image_raw")
        self.declare_parameter("depth_topic", "/wrist_rgbd_depth_sensor/depth/image_raw")
        self.declare_parameter("camera_info_topic", "/wrist_rgbd_depth_sensor/camera_info")
        self.declare_parameter("pointcloud_topic", "/wrist_rgbd_depth_sensor/points")
        self.declare_parameter("target_frame", "base_link")

        # Outputs
        self.declare_parameter("annotated_topic", "/barista_cam_annotated")
        self.declare_parameter("annotated_legacy_topic", "/tray_cam_annotated")
        self.declare_parameter("detection_topic", "/cup_holder_detected")
        self.declare_parameter("marker_topic", "/cup_holder_markers")
        self.declare_parameter("marker_legacy_topic", "/cup_holder_marker")

        # Depth / geometry
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("min_depth_m", 0.02)
        self.declare_parameter("max_depth_m", 3.0)

        # Perception params
        self.declare_parameter("roi_width_ratio", 0.75)
        self.declare_parameter("min_radius_px", 10.0)
        self.declare_parameter("max_radius_px", 20.0)
        self.declare_parameter("max_candidates", 4)

        # Tracker params
        self.declare_parameter("max_ids", 4)
        self.declare_parameter("match_distance_m", 0.06)
        self.declare_parameter("ema_alpha", 0.45)
        self.declare_parameter("min_confirm_frames", 3)
        self.declare_parameter("max_missed_frames", 6)

        # Marker / object geometry defaults
        self.declare_parameter("text_height_offset", 0.10)
        self.declare_parameter("default_hole_height", 0.05)
        self.declare_parameter("min_hole_radius_m", 0.015)
        self.declare_parameter("max_hole_radius_m", 0.050)
        self.declare_parameter("min_cupholder_separation_m", 0.08)

        color_topic = str(self.get_parameter("color_topic").value)
        depth_topic = str(self.get_parameter("depth_topic").value)
        camera_info_topic = str(self.get_parameter("camera_info_topic").value)

        self.target_frame = str(self.get_parameter("target_frame").value)
        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)

        self.text_height_offset = float(self.get_parameter("text_height_offset").value)
        self.default_hole_height = float(self.get_parameter("default_hole_height").value)
        self.min_hole_radius_m = float(self.get_parameter("min_hole_radius_m").value)
        self.max_hole_radius_m = float(self.get_parameter("max_hole_radius_m").value)
        self.min_cupholder_separation_m = float(
            self.get_parameter("min_cupholder_separation_m").value
        )

        p_cfg = PerceptionConfig(
            roi_width_ratio=float(self.get_parameter("roi_width_ratio").value),
            min_radius_px=float(self.get_parameter("min_radius_px").value),
            max_radius_px=float(self.get_parameter("max_radius_px").value),
            max_candidates=int(self.get_parameter("max_candidates").value),
        )
        self.perception = CupholderPerception(p_cfg)

        self.tracker = StableTracker(
            max_ids=int(self.get_parameter("max_ids").value),
            match_distance_m=float(self.get_parameter("match_distance_m").value),
            ema_alpha=float(self.get_parameter("ema_alpha").value),
            min_confirm_frames=int(self.get_parameter("min_confirm_frames").value),
            max_missed_frames=int(self.get_parameter("max_missed_frames").value),
        )

        self.bridge = CvBridge()
        self.k_matrix: Optional[np.ndarray] = None
        self.last_depth_m: Optional[np.ndarray] = None
        self.last_roi_points: Optional[np.ndarray] = None
        self.last_roi_frame: str = ""
        self.last_roi_stamp_sec: float = 0.0
        self.warned_occupancy_frame_mismatch = False
        self.tray_plane: Optional[tuple[np.ndarray, float]] = None
        self.tray_plane_stamp_sec: float = 0.0
        self.rng = np.random.default_rng(42)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.annot_pub = self.create_publisher(
            Image, str(self.get_parameter("annotated_topic").value), 10
        )
        self.annot_legacy_pub = self.create_publisher(
            Image, str(self.get_parameter("annotated_legacy_topic").value), 10
        )
        self.detection_pub = self.create_publisher(
            DetectedObjects, str(self.get_parameter("detection_topic").value), 10
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, str(self.get_parameter("marker_topic").value), 10
        )
        self.marker_legacy_pub = self.create_publisher(
            MarkerArray, str(self.get_parameter("marker_legacy_topic").value), 10
        )

        self.create_subscription(Image, depth_topic, self._depth_cb, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, camera_info_topic, self._camera_info_cb, qos_profile_sensor_data)
        self.create_subscription(Image, color_topic, self._image_cb, qos_profile_sensor_data)
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("pointcloud_topic").value),
            self._pointcloud_cb,
            qos_profile_sensor_data,
        )

        self.get_logger().info("Object detection v2 initialized")

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        if self.k_matrix is None:
            self.k_matrix = np.asarray(msg.k, dtype=float).reshape(3, 3)
            self.get_logger().info("Camera intrinsics loaded")

    def _depth_cb(self, msg: Image) -> None:
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if depth is None:
            return
        if depth.dtype == np.uint16:
            self.last_depth_m = depth.astype(np.float32) * self.depth_scale
        else:
            self.last_depth_m = depth.astype(np.float32)

    def _fit_plane_ransac(self, points_xyz: np.ndarray) -> Optional[tuple[np.ndarray, float]]:
        point_count = points_xyz.shape[0]
        if point_count < 3:
            return None

        best_mask = None
        best_count = 0
        for _ in range(_PLANE_RANSAC_ITERS):
            sample_idx = self.rng.choice(point_count, size=3, replace=False)
            p1, p2, p3 = points_xyz[sample_idx]
            normal = np.cross(p2 - p1, p3 - p1)
            norm = float(np.linalg.norm(normal))
            if norm < 1e-6:
                continue
            normal = normal / norm
            if normal[2] < 0.0:
                normal = -normal

            d = -float(np.dot(normal, p1))
            distances = np.abs(points_xyz @ normal + d)
            mask = distances < _PLANE_DIST_THRESH_M
            inliers = int(np.count_nonzero(mask))
            if inliers > best_count:
                best_count = inliers
                best_mask = mask

        if best_mask is None:
            return None

        inlier_points = points_xyz[best_mask]
        inlier_ratio = float(inlier_points.shape[0]) / float(point_count)
        if inlier_points.shape[0] < _PLANE_MIN_INLIERS or inlier_ratio < _PLANE_MIN_INLIER_RATIO:
            return None

        centroid = np.mean(inlier_points, axis=0)
        _, _, vh = np.linalg.svd(inlier_points - centroid, full_matrices=False)
        normal = vh[-1]
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm < 1e-6:
            return None
        normal = normal / normal_norm
        if normal[2] < 0.0:
            normal = -normal
        if abs(float(normal[2])) < _PLANE_MIN_NZ:
            return None

        d = -float(np.dot(normal, centroid))
        return normal.astype(float), d

    def _pointcloud_cb(self, msg: PointCloud2) -> None:
        raw_points = np.array(
            list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        )
        # Humble can return a structured dtype for read_points(); normalize to Nx3 float32.
        if raw_points.dtype.names is not None:
            points = np.column_stack(
                [raw_points["x"], raw_points["y"], raw_points["z"]]
            ).astype(np.float32, copy=False)
        else:
            points = np.asarray(raw_points, dtype=np.float32).reshape(-1, 3)
        if points.size == 0:
            return

        mask = (
            (points[:, 0] >= _PLANE_ROI_X[0])
            & (points[:, 0] <= _PLANE_ROI_X[1])
            & (points[:, 1] >= _PLANE_ROI_Y[0])
            & (points[:, 1] <= _PLANE_ROI_Y[1])
            & (points[:, 2] >= _PLANE_ROI_Z[0])
            & (points[:, 2] <= _PLANE_ROI_Z[1])
        )
        roi_points = points[mask]
        self.last_roi_points = roi_points
        self.last_roi_frame = msg.header.frame_id
        self.last_roi_stamp_sec = (
            float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        )
        if roi_points.shape[0] < _PLANE_MIN_INLIERS:
            return

        if roi_points.shape[0] > _PLANE_MAX_POINTS:
            idx = self.rng.choice(roi_points.shape[0], size=_PLANE_MAX_POINTS, replace=False)
            roi_points = roi_points[idx]

        plane = self._fit_plane_ransac(roi_points)
        if plane is None:
            return

        self.tray_plane = plane
        self.tray_plane_stamp_sec = self.get_clock().now().nanoseconds * 1e-9

    def _is_occupied_from_pointcloud(
        self, pos_base: np.ndarray, radius_m: float, image_stamp_sec: float
    ) -> bool:
        if self.last_roi_points is None or self.last_roi_points.size == 0:
            return False

        # This occupancy check assumes ROI cloud is in the same frame as detections (target_frame).
        if self.last_roi_frame and self.last_roi_frame != self.target_frame:
            if not self.warned_occupancy_frame_mismatch:
                self.get_logger().warn(
                    "Occupancy check skipped: pointcloud frame '%s' != target_frame '%s'."
                    % (self.last_roi_frame, self.target_frame)
                )
                self.warned_occupancy_frame_mismatch = True
            return False

        cloud_age_sec = abs(image_stamp_sec - self.last_roi_stamp_sec)
        if cloud_age_sec > _OCCUPANCY_MAX_CLOUD_AGE_SEC:
            return False

        points = self.last_roi_points
        xy_radius = float(max(radius_m + _OCCUPANCY_XY_MARGIN_M, 1e-3))

        dx = points[:, 0] - float(pos_base[0])
        dy = points[:, 1] - float(pos_base[1])
        xy_mask = (dx * dx + dy * dy) <= (xy_radius * xy_radius)
        if not np.any(xy_mask):
            return False

        z_threshold = float(pos_base[2]) + _OCCUPANCY_Z_ABOVE_M
        points_above = int(np.count_nonzero(xy_mask & (points[:, 2] > z_threshold)))
        return points_above >= _OCCUPANCY_MIN_POINTS_ABOVE

    def _normalized_positions(self, tracks: List) -> List[np.ndarray]:
        normalized: List[np.ndarray] = []
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        use_plane = (
            self.tray_plane is not None
            and (now_sec - self.tray_plane_stamp_sec) <= _PLANE_MAX_AGE_SEC
        )

        for tr in tracks:
            pos = np.array(tr.position, dtype=float)
            if not use_plane:
                normalized.append(pos)
                continue

            normal, d = self.tray_plane
            if abs(float(normal[2])) < 1e-6:
                normalized.append(pos)
                continue

            z_plane = -(
                float(normal[0]) * float(pos[0])
                + float(normal[1]) * float(pos[1])
                + float(d)
            ) / float(normal[2])
            z_plane += _PLANE_Z_OFFSET_M
            z_new = float(
                np.clip(
                    z_plane,
                    float(pos[2]) - _PLANE_MAX_Z_CORRECTION_M,
                    float(pos[2]) + _PLANE_MAX_Z_CORRECTION_M,
                )
            )
            normalized.append(np.array([float(pos[0]), float(pos[1]), z_new], dtype=float))

        if normalized:
            z_median = float(np.median([p[2] for p in normalized]))
            for p in normalized:
                p[2] = z_median

        return normalized

    def _to_base_link(self, xyz_cam: tuple[float, float, float], source_frame: str) -> Optional[np.ndarray]:
        ps = PointStamped()
        ps.header.frame_id = source_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.point.x, ps.point.y, ps.point.z = xyz_cam

        try:
            tf = self.tf_buffer.lookup_transform(self.target_frame, source_frame, rclpy.time.Time())
            p_out = tf2_geometry_msgs.do_transform_point(ps, tf)
            return np.array([p_out.point.x, p_out.point.y, p_out.point.z], dtype=float)
        except Exception as exc:
            self.get_logger().warn(f"TF transform failed: {exc}")
            return None

    def _apply_min_separation(
        self,
        detections: List[Detection3D],
        candidates: List[Candidate2D],
    ) -> tuple[List[Detection3D], List[Candidate2D]]:
        if len(detections) <= 1:
            return detections, candidates

        order = sorted(
            range(len(detections)),
            key=lambda i: detections[i].score,
            reverse=True,
        )
        kept_by_score: List[int] = []

        for idx in order:
            pos_i = detections[idx].position
            reject = False
            for j in kept_by_score:
                pos_j = detections[j].position
                dist_xy = float(np.linalg.norm(pos_i[:2] - pos_j[:2]))
                min_sep = max(
                    self.min_cupholder_separation_m,
                    0.80 * (detections[idx].radius_m + detections[j].radius_m),
                )
                if dist_xy < min_sep:
                    reject = True
                    break
            if not reject:
                kept_by_score.append(idx)

        kept_set = set(kept_by_score)
        kept_in_input_order = [i for i in range(len(detections)) if i in kept_set]

        filtered_detections = [detections[i] for i in kept_in_input_order]
        filtered_candidates = [candidates[i] for i in kept_in_input_order]
        return filtered_detections, filtered_candidates

    def _publish_markers(self, tracks: List, positions: List[np.ndarray]) -> None:
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        for tr, pos in zip(tracks, positions):
            marker = Marker()
            marker.header.frame_id = self.target_frame
            marker.header.stamp = now
            marker.ns = "fused"
            marker.id = int(tr.track_id)
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.lifetime = Duration(sec=2)
            marker.pose.position.x = float(pos[0])
            marker.pose.position.y = float(pos[1])
            marker.pose.position.z = float(pos[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = float(2.0 * tr.radius_m)
            marker.scale.y = float(2.0 * tr.radius_m)
            marker.scale.z = self.default_hole_height
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.6
            marker_array.markers.append(marker)

            text = Marker()
            text.header.frame_id = self.target_frame
            text.header.stamp = now
            text.ns = "enum"
            text.id = 1000 + int(tr.track_id)
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.lifetime = Duration(sec=2)
            text.pose.position.x = float(pos[0])
            text.pose.position.y = float(pos[1])
            text.pose.position.z = float(pos[2]) + self.text_height_offset
            text.pose.orientation.w = 1.0
            text.scale.z = 0.05
            text.color.r = 0.0
            text.color.g = 0.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = str(int(tr.track_id) + 1)
            marker_array.markers.append(text)

        self.marker_pub.publish(marker_array)
        self.marker_legacy_pub.publish(marker_array)

    def _publish_detections(self, tracks: List, positions: List[np.ndarray]) -> None:
        for tr, pos in zip(tracks, positions):
            msg = DetectedObjects()
            msg.object_id = int(tr.track_id) + 1
            msg.position = Point(
                x=float(pos[0]),
                y=float(pos[1]),
                z=float(pos[2]),
            )
            msg.width = float(2.0 * tr.radius_m)
            msg.thickness = float(2.0 * tr.radius_m)
            msg.height = self.default_hole_height
            self.detection_pub.publish(msg)

    def _annotate(self, bgr: np.ndarray, candidates: List[Candidate2D], track_outputs: List) -> np.ndarray:
        out = bgr.copy()
        id_by_candidate = {}
        for idx, tr in enumerate(track_outputs):
            id_by_candidate[idx] = tr

        for i, cand in enumerate(candidates):
            cv2.circle(out, (cand.u, cand.v), int(round(cand.radius_px)), (0, 0, 255), 2)
            cv2.circle(out, (cand.u, cand.v), 2, (0, 0, 255), -1)

            label = "?"
            color = (180, 180, 180)
            if i in id_by_candidate and id_by_candidate[i].confirmed:
                label = str(id_by_candidate[i].track_id + 1)
                color = (255, 0, 0)

            cv2.putText(
                out,
                label,
                (cand.u + 5, cand.v - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        return out

    def _image_cb(self, msg: Image) -> None:
        if self.k_matrix is None or self.last_depth_m is None:
            return

        image_stamp_sec = (
            float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        )
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        candidates, _ = self.perception.detect(bgr)

        detections_3d: List[Detection3D] = []
        kept_candidates: List[Candidate2D] = []

        fx = float(self.k_matrix[0, 0]) if self.k_matrix is not None else 0.0
        for cand in candidates:
            depth_result = estimate_depth_for_circle(
                self.last_depth_m,
                cand.u,
                cand.v,
                cand.radius_px,
                self.min_depth_m,
                self.max_depth_m,
            )
            if depth_result is None:
                continue

            z, depth_conf = depth_result
            xyz_cam = project_pixel_to_3d(cand.u, cand.v, z, self.k_matrix)
            if xyz_cam is None:
                continue

            xyz_base = self._to_base_link(xyz_cam, msg.header.frame_id)
            if xyz_base is None:
                continue

            if fx <= 1e-6:
                radius_m = 0.035
            else:
                radius_m = float(np.clip(z * cand.radius_px / fx, self.min_hole_radius_m, self.max_hole_radius_m))

            if self._is_occupied_from_pointcloud(xyz_base, radius_m, image_stamp_sec):
                continue

            score = float(np.clip(0.65 * cand.score + 0.35 * depth_conf, 0.0, 1.0))
            detections_3d.append(Detection3D(position=xyz_base, radius_m=radius_m, score=score))
            kept_candidates.append(cand)

        detections_3d, kept_candidates = self._apply_min_separation(
            detections_3d, kept_candidates
        )

        track_outputs = self.tracker.update(detections_3d)
        confirmed_tracks = self.tracker.confirmed_tracks()

        if confirmed_tracks:
            normalized_positions = self._normalized_positions(confirmed_tracks)
            self._publish_markers(confirmed_tracks, normalized_positions)
            self._publish_detections(confirmed_tracks, normalized_positions)

        annotated = self._annotate(bgr, kept_candidates, track_outputs)
        out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        out_msg.header = msg.header
        self.annot_pub.publish(out_msg)
        self.annot_legacy_pub.publish(out_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
