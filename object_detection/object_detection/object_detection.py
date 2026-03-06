#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
import tf2_ros
import tf2_geometry_msgs
from cv_bridge import CvBridge
from rclpy.duration import Duration as RclDuration
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from geometry_msgs.msg import Point, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration
from tf2_ros import ConnectivityException, TransformException
from typing import List, Optional, Tuple

from custom_msgs.msg import DetectedObjects, DetectedSurfaces

try:
    import pcl  # type: ignore
except ImportError:
    pcl = None

ROI_MIN_X, ROI_MAX_X = -0.6, -0.2
ROI_MIN_Y, ROI_MAX_Y = -0.2, 0.2
ROI_MIN_Z, ROI_MAX_Z = -0.75, 0.0

BELOW_BAND = 0.06
Z_GAP_MIN = 0.01
TRAY_RADIUS_CAP = 0.14

CUP_MIN_RADIUS = 0.028
CUP_MAX_RADIUS = 0.04
CUP_MIN_HEIGHT = 0.02
CUP_MAX_HEIGHT = 0.04

CLUSTER_TOLERANCE = 0.04
MIN_CLUSTER_SIZE = 30
MAX_CLUSTER_SIZE = 100000
MIN_CENTROID_DISTANCE = 0.05

OCCUPANCY_Z_MARGIN = 0.04
OCCUPANCY_RADIUS_PAD = 0.01
OCCUPANCY_PTS_THRESH = 20

CUPHOLDER_OFFSET_X = -0.0055
CUPHOLDER_OFFSET_Y = -0.013
CUPHOLDER_OFFSET_Z = 0.01

HOUGH_OFFSET_Y = -0.013
HOUGH_OFFSET_Z = 0.024

FUSION_WEIGHT_HOUGH = 0.3
FUSION_WEIGHT_PCL = 0.7
FUSION_MATCH_THRESHOLD = 0.03

TRAY_MARKER_HEIGHT = 0.09
TEXT_HEIGHT_OFFSET = 0.1


class ObjectDetection(Node):
    def __init__(self) -> None:
        super().__init__("object_detection_node")

        # Optional runtime params (kept minimal)
        self.declare_parameter("pointcloud_topic", "/wrist_rgbd_depth_sensor/points_filtered")
        self.declare_parameter("hough_dp", 1.0)
        self.declare_parameter("hough_min_dist", 60)
        self.declare_parameter("hough_param1", 90)
        self.declare_parameter("hough_param2", 20)
        self.declare_parameter("hough_min_radius", 18)
        self.declare_parameter("hough_max_radius", 24)
        self.declare_parameter("hough_gauss_k", 5)

        pointcloud_topic = str(self.get_parameter("pointcloud_topic").value)

        # Publishers
        self.annot_pub = self.create_publisher(Image, "tray_cam_annotated", 10)
        self.tray_marker_pub = self.create_publisher(MarkerArray, "/tray_marker", 10)
        self.cupholder_marker_pub = self.create_publisher(MarkerArray, "/cup_holder_marker", 10)
        self.tray_detected_pub = self.create_publisher(DetectedSurfaces, "/tray_detected", 10)
        self.cuph_detected_pub = self.create_publisher(DetectedObjects, "/cup_holder_detected", 10)

        # Subscriptions (direct camera/depth Hough branch + pointcloud PCL branch)
        self.create_subscription(Image, "/wrist_rgbd_depth_sensor/image_raw", self.image_callback, 10)
        self.create_subscription(Image, "/wrist_rgbd_depth_sensor/depth/image_raw", self.depth_callback, 10)
        self.create_subscription(CameraInfo, "/wrist_rgbd_depth_sensor/camera_info", self.caminfo_callback, 10)
        self.create_subscription(PointCloud2, pointcloud_topic, self.pc_callback, 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.bridge = CvBridge()
        self.K: Optional[np.ndarray] = None
        self.last_depth: Optional[np.ndarray] = None
        self.depth_scale = 0.001
        self.depth_window = 10
        self.min_depth_m = 0.02
        self.max_depth_m = 3.0

        self.pc_centroids: List[List[float]] = []
        self.pc_dims: List[List[float]] = []
        self.occupied_pcl_centroids: List[List[float]] = []
        self.hough_centroids: List[List[float]] = []

        self.prev_detections: List[Tuple[int, List[float]]] = []
        self._logged_no_pcl_backend = False

        self.get_logger().info("Object detection initialized")

    # ---------- Camera/depth branch ----------
    def caminfo_callback(self, msg: CameraInfo) -> None:
        if self.K is None:
            self.K = np.array(msg.k, dtype=float).reshape(3, 3)
            self.get_logger().info("Camera intrinsics loaded")

    def depth_callback(self, msg: Image) -> None:
        depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if depth_img is None:
            return
        if depth_img.dtype == np.uint16:
            self.last_depth = depth_img.astype(np.float32) * self.depth_scale
        else:
            self.last_depth = depth_img.astype(np.float32)

    def image_callback(self, msg: Image) -> None:
        if self.K is None or self.last_depth is None:
            return

        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        annotated = img.copy()

        h_img, w_img = gray.shape[:2]
        crop_w = 2 * w_img // 3
        proc_gray = gray[:, :crop_w]
        x_off = 0

        k = int(self.get_parameter("hough_gauss_k").value)
        if k % 2 == 0:
            k += 1
        k = max(1, k)
        gray_blur = cv2.GaussianBlur(proc_gray, (k, k), 0) if k > 1 else proc_gray

        circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=float(self.get_parameter("hough_dp").value),
            minDist=float(self.get_parameter("hough_min_dist").value),
            param1=float(self.get_parameter("hough_param1").value),
            param2=float(self.get_parameter("hough_param2").value),
            minRadius=int(self.get_parameter("hough_min_radius").value),
            maxRadius=int(self.get_parameter("hough_max_radius").value),
        )

        hough: List[List[float]] = []
        cam_frame = msg.header.frame_id
        now_msg = self.get_clock().now().to_msg()

        if circles is not None:
            circles = np.around(circles[0]).astype(float)
            for c in circles:
                u_local, v_local = int(round(c[0])), int(round(c[1]))
                r_px = float(c[2])
                u = u_local + x_off
                v = v_local

                cv2.circle(annotated, (u, v), int(round(r_px)), (255, 0, 0), 2)
                cv2.circle(annotated, (u, v), 3, (255, 0, 0), -1)

                xyz = self.pixel_to_3d(u, v)
                if xyz is None:
                    continue

                ps = PointStamped()
                ps.header.frame_id = cam_frame
                ps.header.stamp = now_msg
                ps.point.x, ps.point.y, ps.point.z = xyz

                try:
                    tf = self.tf_buffer.lookup_transform(
                        "base_link", cam_frame, rclpy.time.Time()
                    )
                    ps_out = tf2_geometry_msgs.do_transform_point(ps, tf)
                    ps_out.point.y += HOUGH_OFFSET_Y
                    ps_out.point.z += HOUGH_OFFSET_Z
                    hough.append([ps_out.point.x, ps_out.point.y, ps_out.point.z])
                except Exception as exc:
                    self.get_logger().warn(f"TF transform failed: {exc}")
                    continue

        # Remove holes that look occupied from PCL analysis.
        self.hough_centroids = [
            h
            for h in hough
            if not any(np.linalg.norm(np.asarray(h) - np.asarray(occ)) < 0.04
                       for occ in self.occupied_pcl_centroids)
        ]

        self.publish_cupholders(self.hough_centroids, method="hough")
        fused_labeled = self.fuse_detections()
        self.draw_fused_labels_on_image(annotated, fused_labeled, cam_frame)

        try:
            out = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            out.header = msg.header
            self.annot_pub.publish(out)
        except Exception as exc:
            self.get_logger().warn(f"Failed to publish annotated image: {exc}")

    def pixel_to_3d(self, u: int, v: int) -> Optional[Tuple[float, float, float]]:
        if self.K is None or self.last_depth is None:
            return None

        h, w = self.last_depth.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            return None

        for radius in (self.depth_window, self.depth_window + 2, self.depth_window + 6):
            u0 = max(0, u - radius)
            u1 = min(w - 1, u + radius)
            v0 = max(0, v - radius)
            v1 = min(h - 1, v + radius)
            window = self.last_depth[v0 : v1 + 1, u0 : u1 + 1].flatten()
            window = window[np.isfinite(window) & (window > 1e-6)]
            if window.size == 0:
                continue

            z = float(np.median(window))
            if not (self.min_depth_m <= z <= self.max_depth_m):
                continue

            fx, fy = self.K[0, 0], self.K[1, 1]
            cx, cy = self.K[0, 2], self.K[1, 2]
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            return (x, y, z)

        return None

    # ---------- Pointcloud branch ----------
    def pc_callback(self, msg: PointCloud2) -> None:
        if pcl is None:
            if not self._logged_no_pcl_backend:
                self.get_logger().error(
                    "python-pcl not available. Cupholder detection requires PCL (no numpy fallback)."
                )
                self._logged_no_pcl_backend = True
            return

        cloud = self.from_ros_msg(msg)
        if cloud.size == 0:
            return

        filtered = self.filter_roi(cloud)
        if filtered.size == 0:
            return

        filtered_cloud = pcl.PointCloud()
        filtered_cloud.from_array(filtered.astype(np.float32))

        _, _, tray_cloud = self.extract_plane(filtered_cloud)
        if tray_cloud.size == 0:
            return

        _, surface_centroids, surface_dimensions = self.extract_clusters(tray_cloud, "Tray Cloud")
        if not surface_centroids:
            return

        cupholder_cloud = self.filter_below_surface(filtered_cloud, surface_centroids[0])
        if cupholder_cloud.size == 0:
            self.pc_centroids = []
            self.pc_dims = []
            self.publish_tray(surface_centroids, surface_dimensions)
            return

        centroids, dims = self.detect_cupholders_pcl(cupholder_cloud.to_array(), filtered)

        self.pc_centroids = centroids
        self.pc_dims = dims

        self.publish_tray(surface_centroids, surface_dimensions)
        self.publish_cupholders(self.pc_centroids, self.pc_dims, method="pcl")
        self.fuse_detections()

    def from_ros_msg(self, msg: PointCloud2) -> np.ndarray:
        """Convert PointCloud2 to Nx3 in base_link."""
        try:
            tf = self.tf_buffer.lookup_transform(
                "base_link", msg.header.frame_id, rclpy.time.Time(),
                timeout=RclDuration(seconds=1.0),
            )

            field_offsets = {f.name: f.offset for f in msg.fields}
            if not all(name in field_offsets for name in ("x", "y", "z")):
                self.get_logger().error("PointCloud2 missing x/y/z fields.")
                return np.empty((0, 3), dtype=np.float32)

            n_points = len(msg.data) // msg.point_step
            if n_points <= 0:
                return np.empty((0, 3), dtype=np.float32)

            f32 = np.dtype(">f4") if msg.is_bigendian else np.dtype("<f4")
            stride = (msg.point_step,)
            x = np.ndarray((n_points,), dtype=f32, buffer=msg.data,
                           offset=field_offsets["x"], strides=stride)
            y = np.ndarray((n_points,), dtype=f32, buffer=msg.data,
                           offset=field_offsets["y"], strides=stride)
            z = np.ndarray((n_points,), dtype=f32, buffer=msg.data,
                           offset=field_offsets["z"], strides=stride)

            pts = np.column_stack((x, y, z)).astype(np.float32, copy=False)
            pts = pts[np.isfinite(pts).all(axis=1)]
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
            r = self.quaternion_to_rotation_matrix(q)
            return (pts @ r.T) + t

        except (TransformException, ConnectivityException) as exc:
            self.get_logger().warn(f"Transform lookup failed: {exc}")
            return np.empty((0, 3), dtype=np.float32)
        except Exception as exc:
            self.get_logger().error(f"Error in from_ros_msg: {exc}")
            return np.empty((0, 3), dtype=np.float32)

    @staticmethod
    def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        x, y, z, w = q
        return np.array(
            [
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def filter_roi(points: np.ndarray) -> np.ndarray:
        keep = (
            (points[:, 0] >= ROI_MIN_X) & (points[:, 0] <= ROI_MAX_X)
            & (points[:, 1] >= ROI_MIN_Y) & (points[:, 1] <= ROI_MAX_Y)
            & (points[:, 2] >= ROI_MIN_Z) & (points[:, 2] <= ROI_MAX_Z)
        )
        return points[keep]

    def filter_below_surface(
        self,
        cloud: "pcl.PointCloud",
        surface_centroid: List[float],
        height_threshold: float = BELOW_BAND,
        radius_limit: float = TRAY_RADIUS_CAP,
    ) -> "pcl.PointCloud":
        filtered_indices: List[int] = []
        centroid = surface_centroid

        for i in range(cloud.size):
            point = cloud[i]
            x, y, z = point
            if z < centroid[2] - Z_GAP_MIN and (centroid[2] - z) <= height_threshold:
                distance_from_center = np.sqrt((x - centroid[0]) ** 2 + (y - centroid[1]) ** 2)
                if distance_from_center <= radius_limit:
                    filtered_indices.append(i)

        return cloud.extract(filtered_indices)

    def extract_plane(
        self, cloud: "pcl.PointCloud"
    ) -> Tuple[np.ndarray, np.ndarray, "pcl.PointCloud"]:
        seg = cloud.make_segmenter_normals(ksearch=50)
        seg.set_optimize_coefficients(True)
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_distance_threshold(0.01)
        indices, coefficients = seg.segment()

        if indices is None or len(indices) == 0:
            empty = pcl.PointCloud()
            empty.from_array(np.empty((0, 3), dtype=np.float32))
            return np.array([]), np.array([]), empty

        plane_cloud = cloud.extract(indices)
        return np.array(indices), np.array(coefficients), plane_cloud

    def extract_clusters(
        self, cloud: "pcl.PointCloud", cluster_type: str
    ) -> Tuple[List["pcl.PointCloud"], List[List[float]], List[List[float]]]:
        tree = cloud.make_kdtree()
        ec = cloud.make_EuclideanClusterExtraction()
        ec.set_ClusterTolerance(CLUSTER_TOLERANCE)
        ec.set_MinClusterSize(MIN_CLUSTER_SIZE)
        ec.set_MaxClusterSize(MAX_CLUSTER_SIZE)
        ec.set_SearchMethod(tree)
        cluster_indices = ec.Extract()

        object_clusters: List["pcl.PointCloud"] = []
        cluster_centroids: List[List[float]] = []
        cluster_dimensions: List[List[float]] = []

        for indices in cluster_indices:
            cluster = cloud.extract(indices)
            cluster_arr = cluster.to_array()
            if cluster_arr.shape[0] == 0:
                continue

            centroid = np.mean(cluster_arr, axis=0)
            min_coords = np.min(cluster_arr, axis=0)
            max_coords = np.max(cluster_arr, axis=0)
            dimensions = max_coords - min_coords

            object_clusters.append(cluster)
            cluster_centroids.append(centroid.tolist())
            cluster_dimensions.append(dimensions.tolist())

        if not object_clusters:
            self.get_logger().warning(f"No {cluster_type} clusters extracted...")

        return object_clusters, cluster_centroids, cluster_dimensions

    def estimate_tray(
        self, roi: np.ndarray
    ) -> Tuple[Optional[List[float]], Optional[List[float]], Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
        if roi.shape[0] < 30:
            return None, None, None

        z = roi[:, 2]
        z_hi = float(np.percentile(z, 95))
        top_seed = roi[np.abs(roi[:, 2] - z_hi) <= 0.02]
        if top_seed.shape[0] < 30:
            top_seed = roi

        tray_frame, inlier_top = self.compute_tray_frame_ransac(top_seed)
        if tray_frame is None or inlier_top.shape[0] < 20:
            return None, None, None

        top_uv = self.project_points_to_tray_uv(inlier_top, tray_frame)
        center_uv = np.median(top_uv, axis=0)
        radial_uv = np.linalg.norm(top_uv - center_uv[None, :], axis=1)
        tray_r = float(np.percentile(radial_uv, 90))
        tray_r = float(np.clip(tray_r, 0.08, TRAY_RADIUS_CAP))

        center_xyz = self.tray_uv_to_world(center_uv, tray_frame)
        center = [float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])]
        dims = [2.0 * tray_r, 2.0 * tray_r, TRAY_MARKER_HEIGHT]
        return center, dims, tray_frame

    @staticmethod
    def compute_tray_frame_ransac(
        points: np.ndarray,
    ) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], np.ndarray]:
        if points.shape[0] < 30:
            return None, np.empty((0, 3), dtype=np.float32)

        n_pts = points.shape[0]
        best_count = 0
        best_inliers = None
        rng = np.random.default_rng()

        for _ in range(120):
            idx = rng.choice(n_pts, size=3, replace=False)
            a, b, c = points[idx[0]], points[idx[1]], points[idx[2]]
            n = np.cross(b - a, c - a)
            norm_n = float(np.linalg.norm(n))
            if norm_n < 1e-9:
                continue
            n = n / norm_n
            if n[2] < 0.0:
                n = -n

            d = np.abs((points - a) @ n)
            inliers = d <= 0.003
            count = int(np.sum(inliers))
            if count > best_count:
                best_count = count
                best_inliers = inliers

        if best_inliers is None or best_count < 20:
            return None, np.empty((0, 3), dtype=np.float32)

        inlier_pts = points[best_inliers]
        p0 = np.mean(inlier_pts, axis=0)
        centered = inlier_pts - p0
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        n = eigvecs[:, int(np.argmin(eigvals))]
        n = n / max(1e-9, float(np.linalg.norm(n)))
        if n[2] < 0.0:
            n = -n

        ux = np.asarray([1.0, 0.0, 0.0], dtype=float)
        u = ux - np.dot(ux, n) * n
        if np.linalg.norm(u) < 1e-6:
            uy = np.asarray([0.0, 1.0, 0.0], dtype=float)
            u = uy - np.dot(uy, n) * n
        u = u / max(1e-9, float(np.linalg.norm(u)))
        v = np.cross(n, u)
        v = v / max(1e-9, float(np.linalg.norm(v)))

        frame = (
            np.asarray(p0, dtype=float),
            np.asarray(u, dtype=float),
            np.asarray(v, dtype=float),
            np.asarray(n, dtype=float),
        )
        return frame, inlier_pts

    @staticmethod
    def project_points_to_tray_uv(
        points: np.ndarray,
        tray_frame: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        p0, u, v, _ = tray_frame
        d = points - p0
        return np.column_stack((d @ u, d @ v))

    @staticmethod
    def project_point_to_tray_uv(
        point: np.ndarray,
        tray_frame: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        p0, u, v, _ = tray_frame
        d = point - p0
        return np.asarray([float(np.dot(d, u)), float(np.dot(d, v))], dtype=float)

    @staticmethod
    def tray_uv_to_world(
        uv: np.ndarray,
        tray_frame: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        p0, u, v, _ = tray_frame
        return p0 + float(uv[0]) * u + float(uv[1]) * v

    def extract_below_tray_band(
        self,
        roi: np.ndarray,
        tray_center: List[float],
        tray_frame: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        p0, u, v, n = tray_frame
        d = roi - p0
        uv = np.column_stack((d @ u, d @ v))
        center_uv = self.project_point_to_tray_uv(np.asarray(tray_center, dtype=float), tray_frame)
        radial_uv = np.linalg.norm(uv - center_uv[None, :], axis=1)
        signed_dist = d @ n

        keep = (
            (signed_dist <= -Z_GAP_MIN)
            & (signed_dist >= -BELOW_BAND)
            & (radial_uv <= TRAY_RADIUS_CAP)
        )
        return roi[keep]

    def detect_cupholders_pcl(
        self,
        candidate_points: np.ndarray,
        reference_points: np.ndarray,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        if candidate_points.shape[0] < MIN_CLUSTER_SIZE:
            self.occupied_pcl_centroids = []
            return [], []

        cloud = pcl.PointCloud()
        cloud.from_array(candidate_points.astype(np.float32))

        tree = cloud.make_kdtree()
        ec = cloud.make_EuclideanClusterExtraction()
        ec.set_ClusterTolerance(CLUSTER_TOLERANCE)
        ec.set_MinClusterSize(MIN_CLUSTER_SIZE)
        ec.set_MaxClusterSize(MAX_CLUSTER_SIZE)
        ec.set_SearchMethod(tree)
        cluster_indices = ec.Extract()

        centroids: List[List[float]] = []
        dims_list: List[List[float]] = []
        occupied: List[List[float]] = []

        for indices in cluster_indices:
            cluster = cloud.extract(indices)
            cluster_arr = cluster.to_array()
            if cluster_arr.shape[0] == 0:
                continue

            centroid = np.mean(cluster_arr, axis=0)
            min_coords = np.min(cluster_arr, axis=0)
            max_coords = np.max(cluster_arr, axis=0)
            dims = (max_coords - min_coords).astype(float)

            radius = float(dims[0]) / 2.0
            height = float(dims[2])

            radial = np.hypot(
                reference_points[:, 0] - float(centroid[0]),
                reference_points[:, 1] - float(centroid[1]),
            )
            points_above = np.sum(
                (radial < (radius + OCCUPANCY_RADIUS_PAD))
                & (reference_points[:, 2] > (float(centroid[2]) + OCCUPANCY_Z_MARGIN))
            )
            if int(points_above) > OCCUPANCY_PTS_THRESH:
                occupied.append(centroid.tolist())
                continue

            if not (CUP_MIN_RADIUS <= radius <= CUP_MAX_RADIUS):
                continue
            if not (CUP_MIN_HEIGHT <= height <= CUP_MAX_HEIGHT):
                continue

            centroid_adj = np.asarray(centroid, dtype=float).copy()
            centroid_adj[0] += CUPHOLDER_OFFSET_X
            centroid_adj[1] += CUPHOLDER_OFFSET_Y
            centroid_adj[2] += CUPHOLDER_OFFSET_Z

            centroids.append([float(centroid_adj[0]), float(centroid_adj[1]), float(centroid_adj[2])])
            dims_list.append([float(dims[0]), float(dims[1]), float(dims[2])])

        self.occupied_pcl_centroids = occupied
        return self.filter_close_cupholders_with_dims(centroids, dims_list, MIN_CENTROID_DISTANCE)

    @staticmethod
    def filter_close_cupholders_with_dims(
        centroids: List[List[float]],
        dims: List[List[float]],
        min_distance: float,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        filtered_centroids: List[List[float]] = []
        filtered_dims: List[List[float]] = []

        for centroid, dim in zip(centroids, dims):
            too_close = False
            for existing in filtered_centroids:
                if np.linalg.norm(np.asarray(centroid) - np.asarray(existing)) < min_distance:
                    too_close = True
                    break
            if not too_close:
                filtered_centroids.append(centroid)
                filtered_dims.append(dim)

        return filtered_centroids, filtered_dims

    # ---------- Fusion ----------
    def fuse_detections(self) -> List[Tuple[int, List[float]]]:
        if not self.pc_centroids and not self.hough_centroids:
            return []

        fused_centroids: List[List[float]] = []

        if not self.pc_centroids:
            fused_centroids = self.hough_centroids
        elif not self.hough_centroids:
            fused_centroids = self.pc_centroids
        else:
            for h in self.hough_centroids:
                for p in self.pc_centroids:
                    dist = float(np.linalg.norm(np.asarray(h) - np.asarray(p)))
                    if dist < FUSION_MATCH_THRESHOLD:
                        c = [
                            h_i * FUSION_WEIGHT_HOUGH + p_i * FUSION_WEIGHT_PCL
                            for h_i, p_i in zip(h, p)
                        ]
                        is_occupied = any(
                            np.linalg.norm(np.asarray(c) - np.asarray(occ)) < 0.04
                            for occ in self.occupied_pcl_centroids
                        )
                        if not is_occupied:
                            fused_centroids.append([float(c[0]), float(c[1]), float(c[2])])
                        break

        if fused_centroids:
            return self.publish_cupholders(fused_centroids, method="fused")
        return []

    # ---------- Publishing ----------
    def publish_tray(self, centroids: List[List[float]], dims: List[List[float]]) -> None:
        marker_array = MarkerArray()

        for idx, (centroid, dim) in enumerate(zip(centroids, dims)):
            radius = float(dim[0]) / 2.0
            h = TRAY_MARKER_HEIGHT

            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.id = idx
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = float(centroid[0])
            marker.pose.position.y = float(centroid[1])
            marker.pose.position.z = float(centroid[2]) - h / 2.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = radius * 2.0
            marker.scale.y = radius * 2.0
            marker.scale.z = h
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.4
            marker_array.markers.append(marker)

            msg = DetectedSurfaces()
            msg.surface_id = idx
            msg.position.x = float(centroid[0])
            msg.position.y = float(centroid[1])
            msg.position.z = float(centroid[2])
            msg.height = float(dim[0])
            msg.width = float(dim[1])
            self.tray_detected_pub.publish(msg)

        if marker_array.markers:
            self.tray_marker_pub.publish(marker_array)

    def publish_cupholders(
        self,
        centroids: List[List[float]],
        dims: Optional[List[List[float]]] = None,
        method: str = "fused",
    ) -> List[Tuple[int, List[float]]]:
        if not centroids:
            if method != "fused":
                self.cupholder_marker_pub.publish(MarkerArray())
            return []

        matched = self.match_detections_to_previous(centroids)
        matched = sorted(matched, key=lambda x: x[0])

        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        method_offset = {"hough": 0, "pcl": 100, "fused": 200}.get(method, 0)

        for assigned_id, centroid in matched:
            radius = 0.035
            height = 0.05
            if dims is not None and len(dims) > 0:
                nearest_idx = self.find_nearest_centroid_idx(centroid, centroids)
                if nearest_idx is not None and nearest_idx < len(dims):
                    d = dims[nearest_idx]
                    radius = float(max(d[0], d[1])) / 2.0
                    height = float(d[2])

            marker = Marker()
            marker.ns = method
            marker.header.frame_id = "base_link"
            marker.header.stamp = now
            marker.lifetime = Duration(sec=2)
            marker.id = method_offset + assigned_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = float(centroid[0])
            marker.pose.position.y = float(centroid[1])
            marker.pose.position.z = float(centroid[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = radius * 2.0
            marker.scale.y = radius * 2.0
            marker.scale.z = height
            marker.color.a = 0.6

            if method == "hough":
                marker.pose.position.z += 0.02
                marker.scale.z = 0.003
                marker.color.r, marker.color.g, marker.color.b = (1.0, 1.0, 0.0)
            elif method == "pcl":
                marker.color.r, marker.color.g, marker.color.b = (0.0, 0.0, 1.0)
            else:
                marker.color.r, marker.color.g, marker.color.b = (1.0, 0.0, 0.0)

            marker_array.markers.append(marker)

            if method == "fused":
                text_marker = Marker()
                text_marker.header.frame_id = "base_link"
                text_marker.header.stamp = now
                text_marker.lifetime = Duration(sec=2)
                text_marker.ns = "enum"
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.id = 1000 + assigned_id
                text_marker.text = f"ch_{assigned_id + 1}"
                text_marker.action = Marker.ADD
                text_marker.pose.position.x = float(centroid[0])
                text_marker.pose.position.y = float(centroid[1])
                text_marker.pose.position.z = float(centroid[2]) + TEXT_HEIGHT_OFFSET
                text_marker.pose.orientation.w = 1.0
                text_marker.scale.x = 0.05
                text_marker.scale.y = 0.05
                text_marker.scale.z = 0.05
                text_marker.color.r = 0.0
                text_marker.color.g = 0.0
                text_marker.color.b = 1.0
                text_marker.color.a = 1.0
                marker_array.markers.append(text_marker)

                det = DetectedObjects()
                det.object_id = int(assigned_id + 1)
                det.position = Point(x=float(centroid[0]), y=float(centroid[1]), z=float(centroid[2]))
                det.width = float(radius * 2.0)
                det.thickness = float(radius * 2.0)
                det.height = float(height)
                self.cuph_detected_pub.publish(det)

        self.cupholder_marker_pub.publish(marker_array)
        return matched

    def draw_fused_labels_on_image(
        self,
        image: np.ndarray,
        labeled_fused: List[Tuple[int, List[float]]],
        cam_frame: str,
    ) -> None:
        if self.K is None or not labeled_fused:
            return

        fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
        cx, cy = float(self.K[0, 2]), float(self.K[1, 2])
        h_img, w_img = image.shape[:2]

        for assigned_id, centroid in labeled_fused:
            point_in_cam = self.transform_point(
                centroid[0], centroid[1], centroid[2], "base_link", cam_frame
            )
            if point_in_cam is None:
                continue

            x_c, y_c, z_c = point_in_cam
            if z_c <= 1e-6:
                continue

            u = int(round((fx * x_c / z_c) + cx))
            v = int(round((fy * y_c / z_c) + cy))
            if not (0 <= u < w_img and 0 <= v < h_img):
                continue

            # Compact camera label for better readability in the image panel.
            label = f"ch{assigned_id + 1}"
            cv2.circle(image, (u, v), 4, (255, 0, 0), -1)
            cv2.putText(
                image,
                label,
                (u + 8, v - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

    def transform_point(
        self,
        x: float,
        y: float,
        z: float,
        source_frame: str,
        target_frame: str,
    ) -> Optional[Tuple[float, float, float]]:
        ps = PointStamped()
        ps.header.frame_id = source_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.point.x = float(x)
        ps.point.y = float(y)
        ps.point.z = float(z)
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
            )
            ps_out = tf2_geometry_msgs.do_transform_point(ps, tf)
            return (
                float(ps_out.point.x),
                float(ps_out.point.y),
                float(ps_out.point.z),
            )
        except Exception:
            return None

    @staticmethod
    def find_nearest_centroid_idx(
        target: List[float], candidates: List[List[float]]
    ) -> Optional[int]:
        if not candidates:
            return None
        t = np.asarray(target, dtype=float)
        dists = [float(np.linalg.norm(t - np.asarray(c, dtype=float))) for c in candidates]
        return int(np.argmin(dists))

    def match_detections_to_previous(
        self,
        new_centroids: List[List[float]],
        threshold: float = 0.05,
        max_ids: int = 4,
    ) -> List[Tuple[int, List[float]]]:
        matched: List[Tuple[int, List[float]]] = []
        unmatched: List[List[float]] = []
        used_prev_ids = set()
        id_pool = set(range(max_ids))
        used_current_ids = set()

        for centroid in new_centroids:
            min_dist = float("inf")
            matched_id = None
            for prev_id, prev_pos in self.prev_detections:
                if prev_id in used_prev_ids:
                    continue
                dist = float(np.linalg.norm(np.asarray(centroid) - np.asarray(prev_pos)))
                if dist < threshold and dist < min_dist:
                    matched_id = prev_id
                    min_dist = dist

            if matched_id is not None:
                matched.append((matched_id, centroid))
                used_prev_ids.add(matched_id)
                used_current_ids.add(matched_id)
            else:
                unmatched.append(centroid)

        available_ids = list(id_pool - used_current_ids)
        for centroid in unmatched:
            if not available_ids:
                break
            assigned_id = available_ids.pop(0)
            matched.append((assigned_id, centroid))
            used_current_ids.add(assigned_id)

        self.prev_detections = [(id_, centroid) for id_, centroid in matched]
        return matched


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObjectDetection()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
