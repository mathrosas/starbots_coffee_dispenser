#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

import numpy as np
import pcl
import tf2_ros
from tf2_ros import TransformException, ConnectivityException
from typing import List, Tuple, Union

from custom_msgs.msg import DetectedSurfaces, DetectedObjects


class TrayAndCupholderPerception(Node):
    """
    Detects:
      • Tray (as a plane-ish circular surface) → publishes /tray_marker (green cylinders) and /tray_detected
      • Cup-holders (as cylinders)             → publishes /cup_holder_marker (blue cylinders) and /cup_holder_detected
    Logic updated to:
      1) segment tray plane
      2) derive a "below-tray" band in Z + radial cap around tray centroid
      3) cluster that band and accept clusters by physical dimensions (radius/height)
      4) skip clusters likely occupied by cups (points above surface)
      5) enforce min inter-centroid distance
    """

    def __init__(self) -> None:
        super().__init__('tray_and_cupholder_detection_node')

        # --- Params (tunable) ---
        # Coarse ROI to speed up plane search (meters, base_link)
        self.declare_parameter('roi_min_x', -0.65); self.declare_parameter('roi_max_x', -0.15)
        self.declare_parameter('roi_min_y', -0.30); self.declare_parameter('roi_max_y',  0.30)
        self.declare_parameter('roi_min_z', -0.80); self.declare_parameter('roi_max_z',  0.10)

        # "Below-tray" band selection and radius cap (meters)
        self.declare_parameter('below_band', 0.06)       # how far below tray centroid z we allow
        self.declare_parameter('z_gap_min', 0.01)        # at least this much below tray z (avoid same-plane points)
        self.declare_parameter('tray_radius_cap', 0.14)  # radial cap around tray centroid

        # Cup-holder physical gates (meters)
        self.declare_parameter('cup_min_radius', 0.028)
        self.declare_parameter('cup_max_radius', 0.040)
        self.declare_parameter('cup_min_height', 0.020)
        self.declare_parameter('cup_max_height', 0.040)

        # Clustering parameters
        self.declare_parameter('cluster_tol', 0.04)
        self.declare_parameter('cluster_min', 30)
        self.declare_parameter('cluster_max', 100000)

        # Occupancy check
        self.declare_parameter('occupancy_z_margin', 0.04)  # look for points above holder top by this margin
        self.declare_parameter('occupancy_pts_thresh', 20)   # enough points above → considered occupied

        # De-duplication
        self.declare_parameter('min_centroid_distance', 0.05)

        # Viz tweaks
        self.declare_parameter('tray_marker_height', 0.09)
        self.declare_parameter('cup_marker_height',  0.035)
        self.declare_parameter('text_height_offset', 0.06)

        # --- Subscription (wrist depth cloud) ---
        self.pc_sub_wrist = self.create_subscription(
            PointCloud2, '/wrist_rgbd_depth_sensor/points_filtered',
            self.callback_tray_and_cupholders, 10
        )

        # --- Publishers ---
        self.tray_marker_pub       = self.create_publisher(MarkerArray, '/tray_marker', 10)
        self.cupholder_marker_pub  = self.create_publisher(MarkerArray, '/cup_holder_marker', 10)
        self.tray_detected_pub     = self.create_publisher(DetectedSurfaces, '/tray_detected', 10)
        self.cuph_detected_pub     = self.create_publisher(DetectedObjects,  '/cup_holder_detected', 10)

        # --- TF ---
        self.tf_buffer = tf2_ros.Buffer()
               # Listener must be kept alive by the node
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    # ------------ Core callback ------------
    def callback_tray_and_cupholders(self, msg: PointCloud2) -> None:
        try:
            cloud = self.from_ros_msg(msg)
            if cloud is None or cloud.size == 0:
                return

            # 1) coarse ROI
            roi = self.filter_cloud(
                cloud,
                self.get_parameter('roi_min_x').value, self.get_parameter('roi_max_x').value,
                self.get_parameter('roi_min_y').value, self.get_parameter('roi_max_y').value,
                self.get_parameter('roi_min_z').value, self.get_parameter('roi_max_z').value
            )
            if roi is None or roi.size == 0:
                self.get_logger().warn("Empty ROI after coarse filter.")
                return

            # 2) segment tray plane from ROI, then cluster to get tray centroid/dims
            _, _, tray_plane = self.extract_plane(roi)
            tray_clusters, tray_centroids, tray_dims = self.extract_clusters(
                tray_plane,
                tol=self.get_parameter('cluster_tol').value,
                min_sz=self.get_parameter('cluster_min').value,
                max_sz=self.get_parameter('cluster_max').value
            )
            if not tray_centroids:
                self.get_logger().warn("No tray clusters found on plane.")
                return

            # pick the largest-by-points cluster as tray (simple heuristic)
            tray_main_idx = int(np.argmax([c.size for c in tray_clusters]))
            tray_center   = tray_centroids[tray_main_idx]
            tray_dim_main = tray_dims[tray_main_idx]

            # 3) build a "below-tray" band around tray_center with radial cap
            band = self.filter_below_surface(
                cloud=roi,
                surface_centroid=tray_center,
                z_gap_min=self.get_parameter('z_gap_min').value,
                band_height=self.get_parameter('below_band').value,
                radius_cap=self.get_parameter('tray_radius_cap').value
            )
            if band is None or band.size == 0:
                self.get_logger().warn("No points in below-tray band.")
                # Still publish tray; just no cupholders
                self.publish_tray([tray_center], [tray_dim_main])
                return

            # 4) cluster the band, gate by physical dimensions, occupancy, spacing
            ch_centroids, ch_dims = self.select_cupholder_like_clusters(
                candidate_cloud=band,
                reference_cloud=roi,
                min_r=self.get_parameter('cup_min_radius').value,
                max_r=self.get_parameter('cup_max_radius').value,
                min_h=self.get_parameter('cup_min_height').value,
                max_h=self.get_parameter('cup_max_height').value,
                tol=self.get_parameter('cluster_tol').value,
                min_sz=self.get_parameter('cluster_min').value,
                max_sz=self.get_parameter('cluster_max').value,
                occ_margin=self.get_parameter('occupancy_z_margin').value,
                occ_pts=self.get_parameter('occupancy_pts_thresh').value,
                min_dist=self.get_parameter('min_centroid_distance').value
            )

            # 5) publish everything
            self.publish_tray([tray_center], [tray_dim_main])
            self.publish_cupholders(ch_centroids, ch_dims)

        except Exception as e:
            self.get_logger().error(f"[Tray/Cupholders] Error in callback: {e}")

    # ------------ Conversions / geometry ------------
    def from_ros_msg(self, msg: PointCloud2) -> Union[pcl.PointCloud, None]:
        """Convert PointCloud2 to pcl.PointCloud in base_link frame."""
        try:
            tf = self.tf_buffer.lookup_transform(
                'base_link', msg.header.frame_id,
                rclpy.time.Time(), timeout=Duration(seconds=1.0)
            )
            t = np.array([tf.transform.translation.x,
                          tf.transform.translation.y,
                          tf.transform.translation.z], dtype=np.float32)
            q = np.array([tf.transform.rotation.x,
                          tf.transform.rotation.y,
                          tf.transform.rotation.z,
                          tf.transform.rotation.w], dtype=np.float32)
            R = self.quaternion_to_rotation_matrix(q)

            step = msg.point_step
            n = len(msg.data) // step
            pts = np.empty((n, 3), dtype=np.float32)
            # zero-copy-ish parsing
            for i in range(n):
                s = i * step
                x = np.frombuffer(msg.data[s:s+4],       dtype=np.float32, count=1)[0]
                y = np.frombuffer(msg.data[s+4:s+8],     dtype=np.float32, count=1)[0]
                z = np.frombuffer(msg.data[s+8:s+12],    dtype=np.float32, count=1)[0]
                pts[i, :] = (R @ np.array([x, y, z], dtype=np.float32)) + t

            cloud = pcl.PointCloud()
            cloud.from_array(pts)
            return cloud

        except (TransformException, ConnectivityException) as e:
            self.get_logger().error(f"Transform lookup failed: {e}")
            return None
        except Exception as e:
            self.get_logger().error(f"Error in from_ros_msg: {e}")
            return None

    def quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        x, y, z, w = q
        return np.array([
            [1 - 2*y*y - 2*z*z,   2*x*y - 2*z*w,       2*x*z + 2*y*w],
            [2*x*y + 2*z*w,       1 - 2*x*x - 2*z*z,   2*y*z - 2*x*w],
            [2*x*z - 2*y*w,       2*y*z + 2*x*w,       1 - 2*x*x - 2*y*y]
        ], dtype=np.float32)

    # ------------ Filtering / segmentation / clustering ------------
    def filter_cloud(self, cloud: pcl.PointCloud,
                     min_x: float, max_x: float,
                     min_y: float, max_y: float,
                     min_z: float, max_z: float) -> Union[pcl.PointCloud, None]:
        try:
            keep = []
            for i in range(cloud.size):
                px, py, pz = cloud[i]
                if (min_x <= px <= max_x and min_y <= py <= max_y and min_z <= pz <= max_z):
                    keep.append(i)
            return cloud.extract(keep) if keep else None
        except Exception as e:
            self.get_logger().error(f"Error in filter_cloud: {e}")
            return None

    def extract_plane(self, cloud: pcl.PointCloud) -> Tuple[np.ndarray, np.ndarray, pcl.PointCloud]:
        """RANSAC plane segmentation with normals."""
        seg = cloud.make_segmenter_normals(ksearch=50)
        seg.set_optimize_coefficients(True)
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_distance_threshold(0.01)
        indices, coeffs = seg.segment()
        plane_cloud = cloud.extract(indices) if len(indices) else pcl.PointCloud()
        return indices, coeffs, plane_cloud

    def extract_clusters(self, cloud: pcl.PointCloud, tol: float, min_sz: int, max_sz: int):
        """Euclidean clustering with centroid & AABB dimensions."""
        if cloud is None or cloud.size == 0:
            return [], [], []
        tree = cloud.make_kdtree()
        ec = cloud.make_EuclideanClusterExtraction()
        ec.set_ClusterTolerance(tol)
        ec.set_MinClusterSize(min_sz)
        ec.set_MaxClusterSize(max_sz)
        ec.set_SearchMethod(tree)
        cluster_indices = ec.Extract()

        clusters, centroids, dims = [], [], []
        for ids in cluster_indices:
            c = cloud.extract(ids)
            arr = c.to_array()
            centroid = np.mean(arr, axis=0)
            mn = np.min(arr, axis=0)
            mx = np.max(arr, axis=0)
            clusters.append(c)
            centroids.append(centroid.tolist())
            dims.append((mx - mn).tolist())
        return clusters, centroids, dims

    def filter_below_surface(self,
                             cloud: pcl.PointCloud,
                             surface_centroid: List[float],
                             z_gap_min: float,
                             band_height: float,
                             radius_cap: float) -> Union[pcl.PointCloud, None]:
        """Keep points just below the tray centroid z, within a vertical band and radial cap."""
        keep = []
        cx, cy, cz = surface_centroid
        for i in range(cloud.size):
            x, y, z = cloud[i]
            if z < (cz - z_gap_min) and (cz - z) <= band_height:
                if np.hypot(x - cx, y - cy) <= radius_cap:
                    keep.append(i)
        return cloud.extract(keep) if keep else None

    def select_cupholder_like_clusters(self,
                                       candidate_cloud: pcl.PointCloud,
                                       reference_cloud: pcl.PointCloud,
                                       min_r: float, max_r: float,
                                       min_h: float, max_h: float,
                                       tol: float, min_sz: int, max_sz: int,
                                       occ_margin: float, occ_pts: int,
                                       min_dist: float) -> Tuple[List[List[float]], List[List[float]]]:
        """Cluster candidates, gate by dimensions, skip 'occupied', and de-duplicate centroids."""
        clusters, centroids, dims = self.extract_clusters(candidate_cloud, tol, min_sz, max_sz)

        # Occupancy check helper: look for enough points above centroid within radius+margin
        def looks_occupied(center_xyz: np.ndarray, radius: float) -> bool:
            count = 0
            cx, cy, cz = center_xyz
            z_thresh = cz + occ_margin
            r_thresh = radius + 0.01
            for i in range(reference_cloud.size):
                px, py, pz = reference_cloud[i]
                if pz > z_thresh and np.hypot(px - cx, py - cy) < r_thresh:
                    count += 1
                    if count >= occ_pts:
                        return True
            return False

        accepted_c = []
        accepted_d = []
        for c, d in zip(centroids, dims):
            arr_dim = np.asarray(d, dtype=float)
            # Use planar size as diameter proxy; choose max of x/y
            diameter = float(max(arr_dim[0], arr_dim[1]))
            radius = diameter / 2.0
            height = float(arr_dim[2])

            if not (min_r <= radius <= max_r and min_h <= height <= max_h):
                continue

            if looks_occupied(np.asarray(c, dtype=float), radius):
                self.get_logger().info("Skipping cluster (occupied by cup above).")
                continue

            # de-dup by centroid spacing
            too_close = any(np.linalg.norm(np.asarray(c) - np.asarray(k)) < min_dist for k in accepted_c)
            if too_close:
                continue

            accepted_c.append(c)
            accepted_d.append([diameter, diameter, height])

        return accepted_c, accepted_d

    # ------------ Publishing ------------
    def publish_tray(self, centroids: List[List[float]], dims: List[List[float]]) -> None:
        # Markers (green cylinders)
        ma = MarkerArray()
        h = float(self.get_parameter('tray_marker_height').value)
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
            m.scale.x = m.scale.y = radius * 2.0
            m.scale.z = h
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.0, 0.4
            ma.markers.append(m)

            # publish DetectedSurfaces individually (as your original)
            msg = DetectedSurfaces()
            msg.surface_id = i
            msg.position.x, msg.position.y, msg.position.z = float(c[0]), float(c[1]), float(c[2])
            msg.height, msg.width = float(d[0]), float(d[1])
            self.tray_detected_pub.publish(msg)

        if ma.markers:
            self.tray_marker_pub.publish(ma)

    def publish_cupholders(self, centroids: List[List[float]], dims: List[List[float]]) -> None:
        ma = MarkerArray()
        h = float(self.get_parameter('cup_marker_height').value)
        text_h = float(self.get_parameter('text_height_offset').value)

        for i, (c, d) in enumerate(zip(centroids, dims)):
            diameter = float(max(d[0], d[1]))
            radius = diameter / 2.0

            cyl = Marker()
            cyl.header.frame_id = "base_link"
            cyl.id = i
            cyl.type = Marker.CYLINDER
            cyl.action = Marker.ADD
            cyl.pose.position.x, cyl.pose.position.y, cyl.pose.position.z = float(c[0]), float(c[1]), float(c[2])
            cyl.pose.orientation.w = 1.0
            cyl.scale.x = cyl.scale.y = radius * 2.0
            cyl.scale.z = h
            cyl.color.r, cyl.color.g, cyl.color.b, cyl.color.a = 0.0, 0.0, 1.0, 0.9
            ma.markers.append(cyl)

            txt = Marker()
            txt.header.frame_id = "base_link"
            txt.id = i + 1000
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.text = str(i)
            txt.pose.position.x, txt.pose.position.y, txt.pose.position.z = float(c[0]), float(c[1]), float(c[2]) + text_h
            txt.pose.orientation.w = 1.0
            txt.scale.x = txt.scale.y = txt.scale.z = 0.05
            # CHANGED: make numbers blue like the cylinders (same alpha as cyl)
            txt.color.r, txt.color.g, txt.color.b, txt.color.a = 0.0, 0.0, 1.0, 0.9
            ma.markers.append(txt)

            # publish DetectedObjects individually (as your original)
            m = DetectedObjects()
            m.object_id = i
            m.position = Point(x=float(c[0]), y=float(c[1]), z=float(c[2]))
            m.width = diameter
            m.thickness = diameter
            m.height = float(d[2])  # actual cluster height (or set a fixed one if you prefer)
            self.cuph_detected_pub.publish(m)

        if ma.markers:
            self.cupholder_marker_pub.publish(ma)
        else:
            # publish empty to clear previous markers if needed
            self.cupholder_marker_pub.publish(MarkerArray())

    # ------------ main ------------
def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrayAndCupholderPerception()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
