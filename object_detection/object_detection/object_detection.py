#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
import tf2_ros
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from tf2_ros import ConnectivityException, TransformException
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, TransformStamped
from typing import List, Tuple

from custom_msgs.msg import DetectedObjects, DetectedSurfaces


# =========================
# Hard-coded configuration
# =========================

# Fixed cup pose (in base_link)
CUP_X = 0.296 # 0.300
CUP_Y = 0.334 # 0.330
CUP_Z = 0.035
CUP_FRAME_PARENT = "base_link"
CUP_FRAME_CHILD = "cup"

# Perception ROI (meters, in base_link)
ROI_MIN_X, ROI_MAX_X = -0.65, -0.15
ROI_MIN_Y, ROI_MAX_Y = -0.30, 0.30
ROI_MIN_Z, ROI_MAX_Z = -0.80, 0.20

# Band below tray top used to detect cup-holder rims
BELOW_BAND = 0.08
Z_GAP_MIN = 0.01
TRAY_RADIUS_CAP = 0.14

# Cup-holder physical gates (meters)
CUP_MIN_RADIUS = 0.028
CUP_MAX_RADIUS = 0.040
CUP_MIN_HEIGHT = 0.020
CUP_MAX_HEIGHT = 0.040

# Occupancy check
OCCUPANCY_Z_MARGIN = 0.04
OCCUPANCY_PTS_THRESH = 20

# De-duplication
MIN_CENTROID_DISTANCE = 0.05

# Viz
TRAY_MARKER_HEIGHT = 0.09
CUP_MARKER_HEIGHT = 0.035
TEXT_HEIGHT_OFFSET = 0.06

# Publish per-holder STATIC TFs
PUBLISH_HOLDER_TFS = True
CUP_HOLDER_FRAME_PREFIX = "ch_"


class ObjectDetection(Node):
    """
    OpenCV-based cup-holder detector over wrist depth cloud:
      1) PointCloud2 -> base_link XYZ
      2) ROI filter
      3) Estimate tray top from high-z points
      4) Build below-tray band points
      5) Bird's-eye density map + OpenCV contour circles
      6) Publish cup-holder markers/messages + STATIC TFs ch_#

    Node interface is kept equivalent to the previous implementation:
      - /tray_marker            (MarkerArray)
      - /cup_holder_marker      (MarkerArray)
      - /tray_detected          (DetectedSurfaces)
      - /cup_holder_detected    (DetectedObjects)
      - STATIC TF 'cup' and 'ch_<n>' in 'base_link'
    """

    def __init__(self) -> None:
        super().__init__("object_detection_node")

        # Runtime tuning (kept for compatibility)
        self.declare_parameter("holder_offset_x", 0.00)
        self.declare_parameter("holder_offset_y", 0.00)
        self.declare_parameter("holder_offset_z", 0.00)
        self.declare_parameter("holder_match_max_dist", 0.08)
        self.declare_parameter("log_ordered_holders", False)
        self.declare_parameter("holder_tf_stable_frames", 3)
        self.declare_parameter("holder_smoothing_alpha", 0.25)
        self.declare_parameter("holder_max_jump", 0.03)
        self.declare_parameter("holder_min_separation", 0.05)
        self.declare_parameter("holder_hold_missing_frames", 20)

        # OpenCV detector params
        self.declare_parameter("opencv_map_resolution", 0.003)       # m/px
        self.declare_parameter("opencv_blur_ksize", 7)               # odd
        self.declare_parameter("opencv_morph_kernel_px", 3)
        self.declare_parameter("opencv_min_contour_area_px", 80.0)
        self.declare_parameter("opencv_expected_holes", 4)
        self.declare_parameter("opencv_use_hough", True)
        self.declare_parameter("opencv_hough_dp", 1.2)
        self.declare_parameter("opencv_hough_param1", 80.0)
        self.declare_parameter("opencv_hough_param2", 10.0)
        self.declare_parameter("pointcloud_topic", "/wrist_rgbd_depth_sensor/points")

        pointcloud_topic = str(self.get_parameter("pointcloud_topic").value)

        # Subscription
        self.pc_sub_wrist = self.create_subscription(
            PointCloud2,
            pointcloud_topic,
            self.callback_tray_and_cupholders,
            10,
        )

        # Publishers
        self.tray_marker_pub = self.create_publisher(MarkerArray, "/tray_marker", 10)
        self.cupholder_marker_pub = self.create_publisher(MarkerArray, "/cup_holder_marker", 10)
        self.tray_detected_pub = self.create_publisher(DetectedSurfaces, "/tray_detected", 10)
        self.cuph_detected_pub = self.create_publisher(DetectedObjects, "/cup_holder_detected", 10)

        # TF (listener + static broadcasters)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.holder_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        # Holder STATIC TF gating (keep behavior unchanged)
        self.holders_tf_published = False
        self.last_holder_centroids_raw: List[List[float]] = []
        self.last_holder_centroids: List[List[float]] = []
        self.holder_tf_eps = 0.005  # 5 mm
        self.pending_holder_centroids: List[List[float]] = []
        self.pending_holder_confirmations = 0
        self.waiting_for_tf_logged = False
        self.tracked_holder_centroids: List[List[float]] = []
        self.tracked_holder_dims: List[List[float]] = []
        self.tracked_holder_missing_counts: List[int] = []

        self.publish_static_cup_tf()
        self.get_logger().info(
            f"[object_detection] Published STATIC TF '{CUP_FRAME_CHILD}' in '{CUP_FRAME_PARENT}' "
            f"at ({CUP_X:.3f}, {CUP_Y:.3f}, {CUP_Z:.3f})"
        )

    # --------- Core callback ---------
    def callback_tray_and_cupholders(self, msg: PointCloud2) -> None:
        try:
            cloud = self.from_ros_msg(msg)
            if cloud is None or cloud.shape[0] == 0:
                return

            roi = self.filter_roi(cloud)
            if roi.shape[0] == 0:
                self.get_logger().warn("Empty ROI after coarse filter.")
                return

            tray_center, tray_dim = self.estimate_tray(roi)
            if tray_center is None:
                self.get_logger().warn("Could not estimate tray surface.")
                return

            band = self.extract_below_tray_band(roi, tray_center)
            if band.shape[0] == 0:
                self.get_logger().warn("No points in below-tray band.")
                self.publish_tray([tray_center], [tray_dim])
                return

            ch_centroids, ch_dims = self.detect_cupholders_opencv(
                candidate_points=band,
                reference_points=roi,
                tray_center=tray_center,
            )

            # Keep stable IDs ch_1..ch_n frame to frame
            ch_centroids, ch_dims = self.stable_order_holders(
                ch_centroids, ch_dims, tray_center
            )
            ch_centroids, ch_dims = self.stabilize_holders(
                ch_centroids, ch_dims, tray_center
            )

            if bool(self.get_parameter("log_ordered_holders").value) and ch_centroids:
                ordered = ", ".join(
                    [f"ch_{i+1}=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})" for i, c in enumerate(ch_centroids)]
                )
                self.get_logger().info(f"[HOLDER_ORDER] {ordered}")

            self.publish_tray([tray_center], [tray_dim])
            self.publish_cupholders(ch_centroids, ch_dims)
            self.last_holder_centroids_raw = [c[:] for c in ch_centroids]

        except Exception as e:
            self.get_logger().error(f"[ObjectDetection] Error in callback: {e}")

    # --------- Point cloud / geometry ---------
    def from_ros_msg(self, msg: PointCloud2) -> np.ndarray:
        """Convert PointCloud2 to Nx3 float32 array in base_link."""
        try:
            if not self.tf_buffer.can_transform(
                "base_link",
                msg.header.frame_id,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.0),
            ):
                if not self.waiting_for_tf_logged:
                    self.get_logger().warn(
                        f"Waiting for TF {msg.header.frame_id} -> base_link before processing cloud."
                    )
                    self.waiting_for_tf_logged = True
                return np.empty((0, 3), dtype=np.float32)

            tf = self.tf_buffer.lookup_transform(
                "base_link",
                msg.header.frame_id,
                rclpy.time.Time(),
                timeout=Duration(seconds=1.0),
            )
            self.waiting_for_tf_logged = False

            field_offsets = {f.name: f.offset for f in msg.fields}
            if not all(name in field_offsets for name in ("x", "y", "z")):
                self.get_logger().error("PointCloud2 missing x/y/z fields.")
                return np.empty((0, 3), dtype=np.float32)

            n_points = len(msg.data) // msg.point_step
            if n_points <= 0:
                return np.empty((0, 3), dtype=np.float32)

            f32 = np.dtype(">f4") if msg.is_bigendian else np.dtype("<f4")
            stride = (msg.point_step,)

            x = np.ndarray(
                shape=(n_points,),
                dtype=f32,
                buffer=msg.data,
                offset=field_offsets["x"],
                strides=stride,
            )
            y = np.ndarray(
                shape=(n_points,),
                dtype=f32,
                buffer=msg.data,
                offset=field_offsets["y"],
                strides=stride,
            )
            z = np.ndarray(
                shape=(n_points,),
                dtype=f32,
                buffer=msg.data,
                offset=field_offsets["z"],
                strides=stride,
            )

            pts = np.column_stack((x, y, z)).astype(np.float32, copy=False)
            finite = np.isfinite(pts).all(axis=1)
            pts = pts[finite]
            if pts.size == 0:
                return np.empty((0, 3), dtype=np.float32)

            t = np.array(
                [
                    tf.transform.translation.x,
                    tf.transform.translation.y,
                    tf.transform.translation.z,
                ],
                dtype=np.float32,
            )
            q = np.array(
                [
                    tf.transform.rotation.x,
                    tf.transform.rotation.y,
                    tf.transform.rotation.z,
                    tf.transform.rotation.w,
                ],
                dtype=np.float32,
            )
            R = self.quaternion_to_rotation_matrix(q)
            return (pts @ R.T) + t

        except (TransformException, ConnectivityException) as e:
            if not self.waiting_for_tf_logged:
                self.get_logger().warn(f"Transform lookup failed: {e}")
                self.waiting_for_tf_logged = True
            return np.empty((0, 3), dtype=np.float32)
        except Exception as e:
            self.get_logger().error(f"Error in from_ros_msg: {e}")
            return np.empty((0, 3), dtype=np.float32)

    def quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        x, y, z, w = q
        return np.array(
            [
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
            ],
            dtype=np.float32,
        )

    def filter_roi(self, points: np.ndarray) -> np.ndarray:
        keep = (
            (points[:, 0] >= ROI_MIN_X) & (points[:, 0] <= ROI_MAX_X) &
            (points[:, 1] >= ROI_MIN_Y) & (points[:, 1] <= ROI_MAX_Y) &
            (points[:, 2] >= ROI_MIN_Z) & (points[:, 2] <= ROI_MAX_Z)
        )
        return points[keep]

    def estimate_tray(self, roi: np.ndarray) -> Tuple[List[float], List[float]]:
        """
        Estimate tray center and dimensions from high-z points.
        Returns:
          center [x, y, z]
          dims   [diameter, diameter, height]
        """
        if roi.shape[0] < 20:
            return None, None

        z = roi[:, 2]
        z_min = float(np.min(z))
        z_max = float(np.max(z))
        if z_max - z_min < 1e-4:
            return None, None

        bin_w = 0.005
        bins = max(10, int(np.ceil((z_max - z_min) / bin_w)))
        hist, edges = np.histogram(z, bins=bins, range=(z_min, z_max))
        peak = int(np.argmax(hist))
        tray_z = float(0.5 * (edges[peak] + edges[peak + 1]))

        # Keep only points around dominant tray-z band
        band = max(0.006, 1.5 * bin_w)
        top = roi[(roi[:, 2] >= tray_z - band) & (roi[:, 2] <= tray_z + band)]
        if top.shape[0] < 10:
            return None, None

        cx = float(np.median(top[:, 0]))
        cy = float(np.median(top[:, 1]))
        cz = tray_z

        radial = np.hypot(top[:, 0] - cx, top[:, 1] - cy)
        tray_r = float(np.percentile(radial, 90))
        tray_r = float(np.clip(tray_r, 0.08, TRAY_RADIUS_CAP))

        center = [cx, cy, cz]
        dims = [2.0 * tray_r, 2.0 * tray_r, TRAY_MARKER_HEIGHT]
        return center, dims

    def extract_below_tray_band(self, roi: np.ndarray, tray_center: List[float]) -> np.ndarray:
        cx, cy, cz = tray_center
        dz = cz - roi[:, 2]
        radial = np.hypot(roi[:, 0] - cx, roi[:, 1] - cy)
        keep = (
            (dz >= Z_GAP_MIN) &
            (dz <= BELOW_BAND) &
            (radial <= TRAY_RADIUS_CAP)
        )
        return roi[keep]

    # --------- OpenCV cup-holder detection ---------
    def detect_cupholders_opencv(self,
                                 candidate_points: np.ndarray,
                                 reference_points: np.ndarray,
                                 tray_center: List[float]) -> Tuple[List[List[float]], List[List[float]]]:
        if candidate_points.shape[0] < 25:
            return [], []

        res = float(self.get_parameter("opencv_map_resolution").value)
        half = TRAY_RADIUS_CAP
        x_min, x_max = tray_center[0] - half, tray_center[0] + half
        y_min, y_max = tray_center[1] - half, tray_center[1] + half

        keep = (
            (candidate_points[:, 0] >= x_min) & (candidate_points[:, 0] <= x_max) &
            (candidate_points[:, 1] >= y_min) & (candidate_points[:, 1] <= y_max)
        )
        pts = candidate_points[keep]
        if pts.shape[0] < 20:
            return [], []

        width = int(np.ceil((x_max - x_min) / res)) + 1
        height = int(np.ceil((y_max - y_min) / res)) + 1
        density = np.zeros((height, width), dtype=np.uint16)

        u = np.clip(((pts[:, 0] - x_min) / res).astype(np.int32), 0, width - 1)
        v = np.clip(((pts[:, 1] - y_min) / res).astype(np.int32), 0, height - 1)
        np.add.at(density, (v, u), 1)

        if density.max() <= 0:
            return [], []

        img = ((density.astype(np.float32) / float(density.max())) * 255.0).astype(np.uint8)

        blur_k = int(self.get_parameter("opencv_blur_ksize").value)
        if blur_k < 3:
            blur_k = 3
        if blur_k % 2 == 0:
            blur_k += 1
        img = cv2.GaussianBlur(img, (blur_k, blur_k), 0)

        _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        mk = int(self.get_parameter("opencv_morph_kernel_px").value)
        mk = max(1, mk)
        kernel = np.ones((mk, mk), dtype=np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        min_area_px = float(self.get_parameter("opencv_min_contour_area_px").value)

        accepted_c: List[List[float]] = []
        accepted_d: List[List[float]] = []
        accepted_support: List[int] = []

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_area_px:
                continue

            (cx_px, cy_px), r_px = cv2.minEnclosingCircle(cnt)
            radius = float(r_px) * res
            if not (CUP_MIN_RADIUS <= radius <= CUP_MAX_RADIUS):
                continue

            cx = x_min + float(cx_px) * res
            cy = y_min + float(cy_px) * res

            evaluated = self._evaluate_candidate(
                cx=cx,
                cy=cy,
                radius=radius,
                candidate_points=pts,
                reference_points=reference_points,
                tray_z=float(tray_center[2]),
            )
            if evaluated is None:
                continue
            center_xyz, dim_xyz, support = evaluated
            self._append_or_replace_candidate(
                center_xyz=center_xyz,
                dim_xyz=dim_xyz,
                support=support,
                accepted_c=accepted_c,
                accepted_d=accepted_d,
                accepted_support=accepted_support,
            )

        # Hough fallback helps split merged blobs and recover missing holes.
        if bool(self.get_parameter("opencv_use_hough").value):
            min_r_px = max(3, int(round((CUP_MIN_RADIUS / res) * 0.8)))
            max_r_px = max(min_r_px + 1, int(round((CUP_MAX_RADIUS / res) * 1.2)))
            min_dist_px = max(6, int(round((MIN_CENTROID_DISTANCE / res) * 0.8)))
            circles = cv2.HoughCircles(
                img,
                cv2.HOUGH_GRADIENT,
                dp=float(self.get_parameter("opencv_hough_dp").value),
                minDist=float(min_dist_px),
                param1=float(self.get_parameter("opencv_hough_param1").value),
                param2=float(self.get_parameter("opencv_hough_param2").value),
                minRadius=min_r_px,
                maxRadius=max_r_px,
            )
            if circles is not None:
                for c in np.asarray(circles[0], dtype=np.float32):
                    cx_px, cy_px, r_px = float(c[0]), float(c[1]), float(c[2])
                    cx = x_min + cx_px * res
                    cy = y_min + cy_px * res
                    radius = float(r_px) * res
                    evaluated = self._evaluate_candidate(
                        cx=cx,
                        cy=cy,
                        radius=radius,
                        candidate_points=pts,
                        reference_points=reference_points,
                        tray_z=float(tray_center[2]),
                    )
                    if evaluated is None:
                        continue
                    center_xyz, dim_xyz, support = evaluated
                    self._append_or_replace_candidate(
                        center_xyz=center_xyz,
                        dim_xyz=dim_xyz,
                        support=support,
                        accepted_c=accepted_c,
                        accepted_d=accepted_d,
                        accepted_support=accepted_support,
                    )

        # Keep strongest N detections (default 4 holes).
        expected = max(1, int(self.get_parameter("opencv_expected_holes").value))
        if accepted_c:
            order = np.argsort(np.asarray(accepted_support))[::-1]
            accepted_c = [accepted_c[i] for i in order]
            accepted_d = [accepted_d[i] for i in order]
            if len(accepted_c) > expected:
                accepted_c = accepted_c[:expected]
                accepted_d = accepted_d[:expected]

        return accepted_c, accepted_d

    def _evaluate_candidate(self,
                            cx: float,
                            cy: float,
                            radius: float,
                            candidate_points: np.ndarray,
                            reference_points: np.ndarray,
                            tray_z: float) -> Tuple[np.ndarray, List[float], int] | None:
        radial = np.hypot(candidate_points[:, 0] - cx, candidate_points[:, 1] - cy)
        local = candidate_points[radial <= max(radius, 0.015)]
        support = int(local.shape[0])
        if support < 8:
            return None

        cz = float(np.median(local[:, 2]))
        z05 = float(np.percentile(local[:, 2], 5))
        z95 = float(np.percentile(local[:, 2], 95))
        h = float(np.clip(z95 - z05, CUP_MIN_HEIGHT, CUP_MAX_HEIGHT))
        center = np.asarray([cx, cy, cz], dtype=float)
        if self.is_holder_occupied(center, radius, reference_points, tray_z):
            return None

        diameter = float(radius * 2.0)
        return center, [diameter, diameter, h], support

    def _append_or_replace_candidate(self,
                                     center_xyz: np.ndarray,
                                     dim_xyz: List[float],
                                     support: int,
                                     accepted_c: List[List[float]],
                                     accepted_d: List[List[float]],
                                     accepted_support: List[int]) -> None:
        for i, prev in enumerate(accepted_c):
            if np.linalg.norm(center_xyz - np.asarray(prev, dtype=float)) < MIN_CENTROID_DISTANCE:
                if support > accepted_support[i]:
                    accepted_c[i] = [float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])]
                    accepted_d[i] = dim_xyz
                    accepted_support[i] = support
                return

        accepted_c.append([float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])])
        accepted_d.append(dim_xyz)
        accepted_support.append(support)

    def is_holder_occupied(self,
                           center_xyz: np.ndarray,
                           radius: float,
                           reference_points: np.ndarray,
                           tray_z: float | None = None) -> bool:
        if reference_points.shape[0] == 0:
            return False

        cx, cy, cz = float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])
        z_thr = cz + OCCUPANCY_Z_MARGIN
        # Guard against false "occupied" from tray surface itself.
        if tray_z is not None:
            z_thr = max(z_thr, float(tray_z) + 0.02)
        radial = np.hypot(reference_points[:, 0] - cx, reference_points[:, 1] - cy)
        hits = np.sum((reference_points[:, 2] > z_thr) & (radial <= (radius + 0.01)))
        return bool(hits >= OCCUPANCY_PTS_THRESH)

    # --------- Holder TF gating helpers ---------
    def _centroids_changed(self, centroids: List[List[float]], eps: float) -> bool:
        if len(centroids) != len(self.last_holder_centroids):
            return True
        for a, b in zip(centroids, self.last_holder_centroids):
            if np.linalg.norm(np.asarray(a) - np.asarray(b)) > eps:
                return True
        return False

    def _centroid_sets_close(self,
                             a: List[List[float]],
                             b: List[List[float]],
                             eps: float) -> bool:
        if len(a) != len(b):
            return False
        for ca, cb in zip(a, b):
            if np.linalg.norm(np.asarray(ca) - np.asarray(cb)) > eps:
                return False
        return True

    def stable_order_holders(self,
                             centroids: List[List[float]],
                             dims: List[List[float]],
                             tray_center: List[float]) -> Tuple[List[List[float]], List[List[float]]]:
        if not centroids:
            return [], []

        tray_xy = np.asarray(tray_center[:2], dtype=float)
        pairs = list(zip(centroids, dims))
        pairs.sort(key=lambda p: float(np.arctan2(p[0][1] - tray_xy[1], p[0][0] - tray_xy[0])))

        if not self.last_holder_centroids_raw:
            return [p[0] for p in pairs], [p[1] for p in pairs]

        prev = [np.asarray(c, dtype=float) for c in self.last_holder_centroids_raw]
        curr = [np.asarray(p[0], dtype=float) for p in pairs]
        max_match = float(self.get_parameter("holder_match_max_dist").value)

        used = set()
        ordered_pairs = []
        for p_prev in prev:
            best_i, best_d = -1, float("inf")
            for i, c_now in enumerate(curr):
                if i in used:
                    continue
                d = float(np.linalg.norm(c_now - p_prev))
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_i >= 0 and best_d <= max_match:
                ordered_pairs.append(pairs[best_i])
                used.add(best_i)

        for i, pair in enumerate(pairs):
            if i not in used:
                ordered_pairs.append(pair)

        return [p[0] for p in ordered_pairs], [p[1] for p in ordered_pairs]

    def stabilize_holders(self,
                          centroids: List[List[float]],
                          dims: List[List[float]],
                          tray_center: List[float]) -> Tuple[List[List[float]], List[List[float]]]:
        expected = max(1, int(self.get_parameter("opencv_expected_holes").value))
        alpha = float(np.clip(float(self.get_parameter("holder_smoothing_alpha").value), 0.01, 1.0))
        max_jump = max(0.001, float(self.get_parameter("holder_max_jump").value))
        min_sep = max(0.0, float(self.get_parameter("holder_min_separation").value))
        hold_missing = max(0, int(self.get_parameter("holder_hold_missing_frames").value))
        match_max = float(self.get_parameter("holder_match_max_dist").value)

        if not centroids and not self.tracked_holder_centroids:
            return [], []

        # First-time initialization with currently visible holders.
        if not self.tracked_holder_centroids and centroids:
            init_n = min(len(centroids), expected)
            self.tracked_holder_centroids = [centroids[i][:] for i in range(init_n)]
            self.tracked_holder_dims = [dims[i][:] for i in range(init_n)]
            self.tracked_holder_missing_counts = [0 for _ in range(init_n)]

        prev = [np.asarray(c, dtype=float) for c in self.tracked_holder_centroids]
        prev_dims = [np.asarray(d, dtype=float) for d in self.tracked_holder_dims]
        det = [np.asarray(c, dtype=float) for c in centroids]
        det_dims = [np.asarray(d, dtype=float) for d in dims]

        # Grow track list up to expected count when new detections appear.
        while len(prev) < expected and len(det) > len(prev):
            idx = len(prev)
            prev.append(det[idx].copy())
            prev_dims.append(det_dims[idx].copy())
            self.tracked_holder_missing_counts.append(0)

        n_tracks = len(prev)
        if len(self.tracked_holder_missing_counts) < n_tracks:
            self.tracked_holder_missing_counts += [0] * (n_tracks - len(self.tracked_holder_missing_counts))

        used = set()
        new_centroids: List[np.ndarray] = []
        new_dims: List[np.ndarray] = []
        new_missing = self.tracked_holder_missing_counts[:n_tracks]

        for i in range(n_tracks):
            p = prev[i]
            best_j = -1
            best_d = float("inf")
            for j, c_now in enumerate(det):
                if j in used:
                    continue
                d = float(np.linalg.norm(c_now - p))
                if d < best_d:
                    best_d = d
                    best_j = j

            if best_j >= 0 and best_d <= match_max:
                used.add(best_j)
                meas = det[best_j]
                delta = meas - p
                dist = float(np.linalg.norm(delta))
                # Clamp abrupt frame jumps to avoid marker/TF teleporting.
                if dist > max_jump and dist > 1e-9:
                    meas = p + (delta / dist) * max_jump
                filt = (1.0 - alpha) * p + alpha * meas

                if i < len(prev_dims) and best_j < len(det_dims):
                    dim_f = (1.0 - alpha) * prev_dims[i] + alpha * det_dims[best_j]
                elif i < len(prev_dims):
                    dim_f = prev_dims[i]
                else:
                    dim_f = np.asarray([2.0 * CUP_MIN_RADIUS, 2.0 * CUP_MIN_RADIUS, CUP_MIN_HEIGHT], dtype=float)

                new_centroids.append(filt)
                new_dims.append(dim_f)
                new_missing[i] = 0
            else:
                # Hold last pose for a while if a detection disappears.
                new_centroids.append(p.copy())
                new_dims.append(prev_dims[i] if i < len(prev_dims) else np.asarray(
                    [2.0 * CUP_MIN_RADIUS, 2.0 * CUP_MIN_RADIUS, CUP_MIN_HEIGHT], dtype=float
                ))
                new_missing[i] = new_missing[i] + 1

        # Drop stale tracks only if we have more tracks than expected.
        if len(new_centroids) > expected:
            keep_idx = []
            for i, miss in enumerate(new_missing):
                if miss <= hold_missing:
                    keep_idx.append(i)
            if len(keep_idx) < expected:
                keep_idx = list(range(expected))
            new_centroids = [new_centroids[i] for i in keep_idx[:expected]]
            new_dims = [new_dims[i] for i in keep_idx[:expected]]
            new_missing = [new_missing[i] for i in keep_idx[:expected]]

        # Prevent overlapping holders by pushing pairs apart minimally.
        if min_sep > 1e-6 and len(new_centroids) > 1:
            for _ in range(2):
                for i in range(len(new_centroids)):
                    for j in range(i + 1, len(new_centroids)):
                        v = new_centroids[j] - new_centroids[i]
                        d = float(np.linalg.norm(v))
                        if d >= min_sep:
                            continue
                        if d < 1e-9:
                            v = np.asarray([1.0, 0.0, 0.0], dtype=float)
                            d = 1.0
                        push = 0.5 * (min_sep - d)
                        u = v / d
                        new_centroids[i] = new_centroids[i] - push * u
                        new_centroids[j] = new_centroids[j] + push * u

        # Keep tracks around tray radius limit.
        tray_xy = np.asarray(tray_center[:2], dtype=float)
        for i in range(len(new_centroids)):
            delta_xy = new_centroids[i][:2] - tray_xy
            r = float(np.linalg.norm(delta_xy))
            if r > TRAY_RADIUS_CAP and r > 1e-9:
                new_centroids[i][:2] = tray_xy + (delta_xy / r) * TRAY_RADIUS_CAP

        self.tracked_holder_centroids = [
            [float(c[0]), float(c[1]), float(c[2])] for c in new_centroids
        ]
        self.tracked_holder_dims = [
            [float(d[0]), float(d[1]), float(d[2])] for d in new_dims
        ]
        self.tracked_holder_missing_counts = new_missing
        return self.tracked_holder_centroids, self.tracked_holder_dims

    # --------- Publishing ---------
    def publish_tray(self, centroids: List[List[float]], dims: List[List[float]]) -> None:
        ma = MarkerArray()
        h = float(TRAY_MARKER_HEIGHT)
        for i, (c, d) in enumerate(zip(centroids, dims)):
            radius = float(d[0]) / 2.0

            m = Marker()
            m.header.frame_id = "base_link"
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(c[0])
            m.pose.position.y = float(c[1])
            m.pose.position.z = float(c[2]) - h / 2.0
            m.pose.orientation.w = 1.0
            m.scale.x = radius * 2.0
            m.scale.y = radius * 2.0
            m.scale.z = h
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.0, 0.4
            ma.markers.append(m)

            msg = DetectedSurfaces()
            msg.surface_id = i
            msg.position.x = float(c[0])
            msg.position.y = float(c[1])
            msg.position.z = float(c[2])
            msg.height = float(d[0])
            msg.width = float(d[1])
            self.tray_detected_pub.publish(msg)

        if ma.markers:
            self.tray_marker_pub.publish(ma)

    def publish_cupholders(self, centroids: List[List[float]], dims: List[List[float]]) -> None:
        off_x = float(self.get_parameter("holder_offset_x").value)
        off_y = float(self.get_parameter("holder_offset_y").value)
        off_z = float(self.get_parameter("holder_offset_z").value)

        shifted_centroids = [
            [float(c[0]) + off_x, float(c[1]) + off_y, float(c[2]) + off_z]
            for c in centroids
        ]

        ma = MarkerArray()
        for i, (c, d) in enumerate(zip(shifted_centroids, dims)):
            diameter = float(max(d[0], d[1]))
            radius = diameter / 2.0
            one_based = i + 1

            cyl = Marker()
            cyl.header.frame_id = "base_link"
            cyl.id = i
            cyl.type = Marker.CYLINDER
            cyl.action = Marker.ADD
            cyl.pose.position.x = float(c[0])
            cyl.pose.position.y = float(c[1])
            cyl.pose.position.z = float(c[2])
            cyl.pose.orientation.w = 1.0
            cyl.scale.x = radius * 2.0
            cyl.scale.y = radius * 2.0
            cyl.scale.z = CUP_MARKER_HEIGHT
            cyl.color.r, cyl.color.g, cyl.color.b, cyl.color.a = 0.0, 0.0, 1.0, 0.9
            ma.markers.append(cyl)

            txt = Marker()
            txt.header.frame_id = "base_link"
            txt.id = i + 1000
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.text = f"ch_{one_based}"
            txt.pose.position.x = float(c[0])
            txt.pose.position.y = float(c[1])
            txt.pose.position.z = float(c[2]) + TEXT_HEIGHT_OFFSET
            txt.pose.orientation.w = 1.0
            txt.scale.x = 0.05
            txt.scale.y = 0.05
            txt.scale.z = 0.05
            txt.color.r = 0.0
            txt.color.g = 0.0
            txt.color.b = 1.0
            txt.color.a = 0.95
            ma.markers.append(txt)

            msg = DetectedObjects()
            msg.object_id = one_based
            msg.position = Point(x=float(c[0]), y=float(c[1]), z=float(c[2]))
            msg.width = diameter
            msg.thickness = diameter
            msg.height = float(d[2])
            self.cuph_detected_pub.publish(msg)

        self.cupholder_marker_pub.publish(ma)

        # STATIC TFs: publish once, then update only when change is stable.
        if PUBLISH_HOLDER_TFS and shifted_centroids:
            # Ignore temporary partial detections (e.g. 2 -> 1 -> 2 flicker).
            if self.holders_tf_published and len(shifted_centroids) < len(self.last_holder_centroids):
                return

            changed = (not self.holders_tf_published) or self._centroids_changed(
                shifted_centroids, self.holder_tf_eps
            )
            if not changed:
                self.pending_holder_centroids = []
                self.pending_holder_confirmations = 0
                return

            if (
                not self.pending_holder_centroids
                or not self._centroid_sets_close(
                    shifted_centroids, self.pending_holder_centroids, self.holder_tf_eps
                )
            ):
                self.pending_holder_centroids = [c[:] for c in shifted_centroids]
                self.pending_holder_confirmations = 1
            else:
                self.pending_holder_confirmations += 1

            stable_frames = max(1, int(self.get_parameter("holder_tf_stable_frames").value))
            if self.pending_holder_confirmations < stable_frames:
                return

            self.publish_static_cupholder_tfs(self.pending_holder_centroids)
            self.holders_tf_published = True
            self.last_holder_centroids = [c[:] for c in self.pending_holder_centroids]
            self.pending_holder_centroids = []
            self.pending_holder_confirmations = 0
            self.get_logger().info("[TF] Cup-holder STATIC TFs updated (gated).")

    # --------- Static TF helpers ---------
    def publish_static_cup_tf(self) -> None:
        ts = TransformStamped()
        ts.header.stamp = self.get_clock().now().to_msg()
        ts.header.frame_id = CUP_FRAME_PARENT
        ts.child_frame_id = CUP_FRAME_CHILD
        ts.transform.translation.x = CUP_X
        ts.transform.translation.y = CUP_Y
        ts.transform.translation.z = CUP_Z
        ts.transform.rotation.w = 1.0
        self.static_broadcaster.sendTransform(ts)

    def publish_static_cupholder_tfs(self, centroids: List[List[float]]) -> None:
        if not centroids:
            return

        now = self.get_clock().now().to_msg()
        tfs: List[TransformStamped] = []
        for i, c in enumerate(centroids):
            one_based = i + 1
            ts = TransformStamped()
            ts.header.stamp = now
            ts.header.frame_id = CUP_FRAME_PARENT
            ts.child_frame_id = f"{CUP_HOLDER_FRAME_PREFIX}{one_based}"
            ts.transform.translation.x = float(c[0])
            ts.transform.translation.y = float(c[1])
            ts.transform.translation.z = float(c[2])
            ts.transform.rotation.w = 1.0
            tfs.append(ts)

        self.holder_static_broadcaster.sendTransform(tfs)
        self.get_logger().info(f"Published {len(tfs)} STATIC TFs for cup-holders (ch_#).")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObjectDetection()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
