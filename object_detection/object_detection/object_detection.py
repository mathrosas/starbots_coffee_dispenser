#!/usr/bin/env python3

from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
import tf2_geometry_msgs
import tf2_ros
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PointStamped
from rclpy.duration import Duration as RclDuration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from tf2_ros import ConnectivityException, TransformException
from visualization_msgs.msg import Marker, MarkerArray

from custom_msgs.msg import DetectedObjects, DetectedSurfaces

try:
    import pcl  # type: ignore
except ImportError:
    pcl = None


class ObjectDetection(Node):
    def __init__(self) -> None:
        super().__init__("object_detection_node")

        self.declare_parameter("pointcloud_topic", "/D415/barista_points")
        pointcloud_topic = str(self.get_parameter("pointcloud_topic").value)

        self.annot_pub = self.create_publisher(Image, "barista_cam_annotated", 10)
        self.annot_legacy_pub = self.create_publisher(Image, "tray_cam_annotated", 10)
        self.tray_detected_pub = self.create_publisher(DetectedSurfaces, "/tray_detected", 10)
        self.tray_marker_pub = self.create_publisher(MarkerArray, "/tray_marker", 10)
        self.cupholder_marker_pub = self.create_publisher(MarkerArray, "/cup_holder_markers", 10)
        self.cupholder_marker_legacy_pub = self.create_publisher(MarkerArray, "/cup_holder_marker", 10)
        self.cuph_detected_pub = self.create_publisher(DetectedObjects, "/cup_holder_detected", 10)

        self.create_subscription(
            Image, "/D415/color/image_raw", self.image_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            Image,
            "/D415/aligned_depth_to_color/image_raw",
            self.depth_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            "/D415/aligned_depth_to_color/camera_info",
            self.caminfo_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointCloud2, pointcloud_topic, self.pc_callback, qos_profile_sensor_data
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.bridge = CvBridge()
        self.pc_centroids: List[List[float]] = []
        self.occupied_pcl_centroids: List[List[float]] = []
        self.hough_centroids: List[List[float]] = []
        self.prev_detections: List[Tuple[int, List[float]]] = []
        self.K: Optional[np.ndarray] = None
        self.last_depth: Optional[np.ndarray] = None
        self.depth_scale = 0.001
        self.depth_window = 10
        self.min_depth_m = 0.02
        self.max_depth_m = 3.0

        self.weight_hough = 0.4
        self.weight_pcl = 0.6
        self.fuse_threshold = 0.03

        self.dp = 1.6
        self.minDist = 20
        self.param1 = 162
        self.param2 = 30
        self.minRadius = 10
        self.maxRadius = 18
        self.gauss_k = 6

        self.get_logger().info("Object detection initialized")

    # ---------- Camera / depth ----------
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
        crop_w = 3 * w_img // 4
        proc_gray = gray[:, :crop_w]

        k = self.gauss_k if (self.gauss_k % 2 == 1) else self.gauss_k + 1
        gray_blur = cv2.GaussianBlur(proc_gray, (k, k), 0) if k > 1 else proc_gray

        circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            self.dp,
            self.minDist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.minRadius,
            maxRadius=self.maxRadius,
        )

        self.hough_centroids = []
        cam_frame = msg.header.frame_id
        now_msg = self.get_clock().now().to_msg()

        if circles is not None:
            circles = np.around(circles[0]).astype(float)
            for c in circles:
                u_local, v_local = int(round(c[0])), int(round(c[1]))
                r_px = float(c[2])
                u = u_local
                v = v_local

                cv2.circle(annotated, (u, v), int(round(r_px)), (0, 255, 0), 2)
                cv2.circle(annotated, (u, v), 3, (0, 0, 255), -1)

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
                    self.hough_centroids.append(
                        [ps_out.point.x, ps_out.point.y, ps_out.point.z]
                    )
                except Exception as exc:
                    self.get_logger().warn(f"TF transform failed: {exc}")
                    continue

        self.hough_centroids = [
            h
            for h in self.hough_centroids
            if not any(
                np.linalg.norm(np.array(h) - np.array(occ)) < 0.04
                for occ in self.occupied_pcl_centroids
            )
        ]

        try:
            out = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            out.header = msg.header
            self.annot_pub.publish(out)
            self.annot_legacy_pub.publish(out)
        except Exception as exc:
            self.get_logger().warn(f"Failed to publish annotated image: {exc}")

        self.publish_cupholders(self.hough_centroids, method="hough")
        self.fuse_detections()

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

    # ---------- Fusion ----------
    def fuse_detections(self) -> None:
        if not self.pc_centroids and not self.hough_centroids:
            return

        fused: List[List[float]] = []
        if not self.pc_centroids:
            fused = self.hough_centroids
        elif not self.hough_centroids:
            fused = self.pc_centroids
        else:
            for h in self.hough_centroids:
                for p in self.pc_centroids:
                    dist = np.linalg.norm(np.array(h) - np.array(p))
                    if dist < self.fuse_threshold:
                        wavg = [
                            (h_i * self.weight_hough + p_i * self.weight_pcl)
                            for h_i, p_i in zip(h, p)
                        ]
                        is_occupied = any(
                            np.linalg.norm(np.array(wavg) - np.array(occ)) < 0.04
                            for occ in self.occupied_pcl_centroids
                        )
                        if not is_occupied:
                            fused.append(wavg)
                        break

        if fused:
            self.publish_cupholders(fused, method="fused")

    # ---------- Pointcloud ----------
    def pc_callback(self, msg: PointCloud2) -> None:
        if pcl is None:
            self.get_logger().error(
                "python-pcl not available. Cupholder detection requires PCL."
            )
            return

        try:
            self.occupied_pcl_centroids = []
            cloud = self.from_ros_msg(msg)
            if cloud is None or cloud.size == 0:
                return

            filtered_cloud = self.filter_cloud(
                cloud,
                min_x=-0.7,
                max_x=-0.2,
                min_y=-0.2,
                max_y=0.5,
                min_z=-0.6,
                max_z=-0.3,
            )
            if filtered_cloud is None or filtered_cloud.size == 0:
                return

            _, _, tray_cloud = self.extract_plane(filtered_cloud)
            if tray_cloud is None or tray_cloud.size == 0:
                return

            _, surface_centroids, surface_dimensions = self.extract_clusters(
                tray_cloud, "Tray Cloud"
            )
            if not surface_centroids:
                return

            cupholder_cloud = self.filter_below_surface(filtered_cloud, surface_centroids[0])
            cup_holder_centroids, cup_holder_dimensions = self.extract_cylinders(
                cupholder_cloud, filtered_cloud
            )

            self.pc_centroids = cup_holder_centroids
            self.publish_tray(surface_centroids, surface_dimensions)
            self.publish_cupholders(
                cup_holder_centroids,
                dimensions=cup_holder_dimensions,
                method="pcl",
            )

        except Exception as exc:
            self.get_logger().error(f"Error in pc_callback: {exc}")

    def from_ros_msg(self, msg: PointCloud2) -> Optional["pcl.PointCloud"]:
        try:
            transform = self.tf_buffer.lookup_transform(
                "base_link",
                msg.header.frame_id,
                rclpy.time.Time(),
                timeout=RclDuration(seconds=1.0),
            )

            field_offsets = {f.name: f.offset for f in msg.fields}
            if not all(name in field_offsets for name in ("x", "y", "z")):
                self.get_logger().error("PointCloud2 missing x/y/z fields")
                return None

            n_points = len(msg.data) // msg.point_step
            if n_points <= 0:
                return None

            f32 = np.dtype(">f4") if msg.is_bigendian else np.dtype("<f4")
            stride = (msg.point_step,)
            x = np.ndarray(
                (n_points,),
                dtype=f32,
                buffer=msg.data,
                offset=field_offsets["x"],
                strides=stride,
            )
            y = np.ndarray(
                (n_points,),
                dtype=f32,
                buffer=msg.data,
                offset=field_offsets["y"],
                strides=stride,
            )
            z = np.ndarray(
                (n_points,),
                dtype=f32,
                buffer=msg.data,
                offset=field_offsets["z"],
                strides=stride,
            )

            points = np.column_stack((x, y, z)).astype(np.float32, copy=False)
            points = points[np.isfinite(points).all(axis=1)]
            if points.size == 0:
                return None

            translation = np.array(
                [
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                ],
                dtype=np.float32,
            )
            rotation_quaternion = np.array(
                [
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w,
                ],
                dtype=np.float32,
            )
            rotation_matrix = self.quaternion_to_rotation_matrix(rotation_quaternion)
            points_base = (points @ rotation_matrix.T) + translation

            cloud = pcl.PointCloud()
            cloud.from_array(points_base.astype(np.float32))
            return cloud

        except (TransformException, ConnectivityException) as exc:
            self.get_logger().error(f"Transform lookup failed: {exc}")
            return None
        except Exception as exc:
            self.get_logger().error(f"Error in from_ros_msg: {exc}")
            return None

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
    def filter_cloud(
        cloud: "pcl.PointCloud",
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        min_z: float,
        max_z: float,
    ) -> "pcl.PointCloud":
        indices = []
        for i in range(cloud.size):
            pt = cloud[i]
            if (
                min_x <= pt[0] <= max_x
                and min_y <= pt[1] <= max_y
                and min_z <= pt[2] <= max_z
            ):
                indices.append(i)
        return cloud.extract(indices)

    @staticmethod
    def filter_below_surface(
        cloud: "pcl.PointCloud",
        surface_centroid: List[float],
        height_threshold: float = 0.06,
        radius_limit: float = 0.15,
    ) -> "pcl.PointCloud":
        filtered_indices = []

        for i in range(cloud.size):
            x, y, z = cloud[i]
            if z < surface_centroid[2] - 0.02 and (surface_centroid[2] - z) <= height_threshold:
                distance_from_center = np.sqrt(
                    (x - surface_centroid[0]) ** 2 + (y - surface_centroid[1]) ** 2
                )
                if distance_from_center <= radius_limit:
                    filtered_indices.append(i)

        return cloud.extract(filtered_indices)

    @staticmethod
    def filter_close_cupholder(
        centroids: List[List[float]], min_distance: float
    ) -> List[List[float]]:
        filtered_centroids: List[List[float]] = []

        for centroid in centroids:
            too_close = False
            for existing in filtered_centroids:
                distance = np.linalg.norm(np.array(centroid) - np.array(existing))
                if distance < min_distance:
                    too_close = True
                    break
            if not too_close:
                filtered_centroids.append(centroid)

        return filtered_centroids

    @staticmethod
    def extract_plane(
        cloud: "pcl.PointCloud",
    ) -> Tuple[np.ndarray, np.ndarray, "pcl.PointCloud"]:
        seg = cloud.make_segmenter_normals(ksearch=50)
        seg.set_optimize_coefficients(True)
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_distance_threshold(0.01)
        indices, coefficients = seg.segment()

        plane_cloud = cloud.extract(indices)
        return np.asarray(indices), np.asarray(coefficients), plane_cloud

    def extract_clusters(
        self, cloud: "pcl.PointCloud", cluster_type: str
    ) -> Tuple[List["pcl.PointCloud"], List[List[float]], List[List[float]]]:
        tree = cloud.make_kdtree()
        ec = cloud.make_EuclideanClusterExtraction()
        ec.set_ClusterTolerance(0.04)
        ec.set_MinClusterSize(30)
        ec.set_MaxClusterSize(100000)
        ec.set_SearchMethod(tree)

        cluster_indices = ec.Extract()

        object_clusters: List["pcl.PointCloud"] = []
        cluster_centroids: List[List[float]] = []
        cluster_dimensions: List[List[float]] = []

        for idx, indices in enumerate(cluster_indices):
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

            self.get_logger().info(
                f"\\n=============================================\\n"
                f"Plane Cluster {idx + 1} has {len(indices)} points:\\n"
                f"Centroid of cluster {idx + 1}: {centroid}\\n"
                f"Dimensions of {cluster_type} cluster {idx + 1}: {dimensions}"
            )

        if not object_clusters:
            self.get_logger().warning(f"No {cluster_type} clusters extracted...")

        return object_clusters, cluster_centroids, cluster_dimensions

    def extract_cylinders(
        self,
        cupholder_cloud: "pcl.PointCloud",
        filtered_cloud: "pcl.PointCloud",
        min_distance: float = 0.05,
        min_height: float = 0.005,
        max_height: float = 0.04,
        min_radius: float = 0.01,
        max_radius: float = 0.04,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        tree = cupholder_cloud.make_kdtree()
        ec = cupholder_cloud.make_EuclideanClusterExtraction()
        ec.set_ClusterTolerance(0.04)
        ec.set_MinClusterSize(30)
        ec.set_MaxClusterSize(100000)
        ec.set_SearchMethod(tree)
        cluster_indices = ec.Extract()

        cupholder_centroids: List[List[float]] = []
        cupholder_dimensions: List[List[float]] = []

        for idx, indices in enumerate(cluster_indices):
            cluster = cupholder_cloud.extract(indices)
            cluster_arr = cluster.to_array()
            if cluster_arr.shape[0] == 0:
                continue

            centroid = np.mean(cluster_arr, axis=0)
            min_coords = np.min(cluster_arr, axis=0)
            max_coords = np.max(cluster_arr, axis=0)
            dimensions = max_coords - min_coords
            radius = dimensions[0] / 2.0
            height = dimensions[2]

            points_above = []
            for i in range(filtered_cloud.size):
                pt = filtered_cloud[i]
                xy_dist = np.linalg.norm(pt[:2] - centroid[:2])
                if xy_dist < radius + 0.01 and pt[2] > centroid[2] + height / 2.0 + 0.03:
                    points_above.append(pt)
            if len(points_above) > 10:
                self.occupied_pcl_centroids.append(centroid.tolist())
                self.get_logger().info(
                    f"Skipping cluster {idx + 1}: likely occupied by a cup."
                )
                continue

            if min_radius <= radius <= max_radius and min_height <= height <= max_height:
                cupholder_centroids.append(centroid.tolist())
                cupholder_dimensions.append(dimensions.tolist())

            self.get_logger().info(
                f"\\n=============================================\\n"
                f"Cluster {idx + 1} has {len(indices)} points:\\n"
                f"Centroid of cluster {idx + 1}: {centroid}\\n"
                f"Radius of cluster {idx + 1}: {radius}\\n"
                f"Height of cluster {idx + 1}: {height}"
            )

        if not cupholder_centroids:
            self.get_logger().warning("No cupholder-like clusters detected!")

        cupholder_centroids = self.filter_close_cupholder(cupholder_centroids, min_distance)

        sorted_results = sorted(
            zip(cupholder_centroids, cupholder_dimensions), key=lambda x: x[0][0]
        )
        if sorted_results:
            cupholder_centroids, cupholder_dimensions = map(list, zip(*sorted_results))
        else:
            cupholder_centroids, cupholder_dimensions = [], []

        return cupholder_centroids, cupholder_dimensions

    # ---------- Publishing ----------
    def publish_tray(
        self, surface_centroids: List[List[float]], surface_dimensions: List[List[float]]
    ) -> None:
        marker_array = MarkerArray()

        for idx, (centroid, dimensions) in enumerate(
            zip(surface_centroids, surface_dimensions)
        ):
            radius = float(dimensions[0]) / 2.0
            height = 0.09

            cylinder_marker = Marker()
            cylinder_marker.header.frame_id = "base_link"
            cylinder_marker.id = idx
            cylinder_marker.type = Marker.CYLINDER
            cylinder_marker.action = Marker.ADD
            cylinder_marker.pose.position.x = float(centroid[0])
            cylinder_marker.pose.position.y = float(centroid[1]) + 0.018
            cylinder_marker.pose.position.z = float(centroid[2]) - height / 2.0
            cylinder_marker.pose.orientation.w = 1.0
            cylinder_marker.scale.x = radius * 2.0
            cylinder_marker.scale.y = radius * 2.0
            cylinder_marker.scale.z = height
            cylinder_marker.color.r = 0.0
            cylinder_marker.color.g = 1.0
            cylinder_marker.color.b = 0.0
            cylinder_marker.color.a = 0.4
            marker_array.markers.append(cylinder_marker)

            surface_msg = DetectedSurfaces()
            surface_msg.surface_id = int(idx)
            surface_msg.position.x = float(centroid[0])
            surface_msg.position.y = float(centroid[1]) + 0.018
            surface_msg.position.z = float(centroid[2])
            surface_msg.height = float(dimensions[0])
            surface_msg.width = float(dimensions[1])
            self.tray_detected_pub.publish(surface_msg)

        self.tray_marker_pub.publish(marker_array)

    def publish_cupholders(
        self,
        centroids: List[List[float]],
        dimensions: Optional[List[List[float]]] = None,
        method: str = "fused",
    ) -> None:
        matched_centroids = self.match_detections_to_previous(centroids)

        x_values = [c[0] for _, c in matched_centroids]
        if not x_values:
            self.get_logger().warn(
                f"No matched centroids found for method '{method}'. Skipping publish."
            )
            return
        x_center = sum(x_values) / len(x_values)
        correction_factor = 0.1

        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()
        text_height_offset = 0.1

        for assigned_id, centroid in matched_centroids:
            radius = 0.035
            height = 0.05
            if dimensions and assigned_id < len(dimensions):
                radius = float(dimensions[assigned_id][0]) / 2.0
                height = float(dimensions[assigned_id][1])

            centroid[0] += 0.005
            centroid[1] -= 0.002

            marker = Marker()
            marker.ns = method
            marker.header.frame_id = "base_link"
            marker.header.stamp = now
            marker.lifetime = Duration(sec=2)
            marker.id = assigned_id
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
                marker.pose.position.z += 0.025
                marker.scale.z = 0.003
                marker.color.r, marker.color.g, marker.color.b = (1.0, 1.0, 0.0)
            elif method == "pcl":
                marker.color.r, marker.color.g, marker.color.b = (0.0, 0.0, 1.0)
            elif method == "fused":
                marker.color.r, marker.color.g, marker.color.b = (1.0, 0.0, 0.0)

                text_marker = Marker()
                text_marker.header.frame_id = "base_link"
                text_marker.header.stamp = now
                text_marker.ns = "enum"
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.id = 1000 + assigned_id
                text_marker.text = f"ch_{assigned_id + 1}"
                text_marker.lifetime = Duration(sec=2)
                text_marker.action = Marker.ADD
                text_marker.pose.position.x = float(centroid[0])
                text_marker.pose.position.y = float(centroid[1])
                text_marker.pose.position.z = float(centroid[2]) + text_height_offset
                text_marker.pose.orientation.w = 1.0
                text_marker.scale.x = 0.05
                text_marker.scale.y = 0.05
                text_marker.scale.z = 0.05
                text_marker.color.r = 0.0
                text_marker.color.g = 0.0
                text_marker.color.b = 1.0
                text_marker.color.a = 1.0
                marker_array.markers.append(text_marker)

                adjusted_x = centroid[0] - (centroid[0] - x_center) * correction_factor
                det = DetectedObjects()
                det.object_id = int(assigned_id + 1)
                det.position = Point(
                    x=float(adjusted_x),
                    y=float(centroid[1]),
                    z=float(centroid[2]),
                )
                det.width = float(radius * 2.0)
                det.thickness = float(radius * 2.0)
                det.height = float(height)
                self.cuph_detected_pub.publish(det)

            marker_array.markers.append(marker)

        self.cupholder_marker_pub.publish(marker_array)
        self.cupholder_marker_legacy_pub.publish(marker_array)

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
                dist = np.linalg.norm(np.array(centroid) - np.array(prev_pos))
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
