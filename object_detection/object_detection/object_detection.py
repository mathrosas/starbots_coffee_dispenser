#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np
import pcl
import tf2_ros
from tf2_ros import TransformException, ConnectivityException
from typing import List, Tuple, Union

from custom_msgs.msg import DetectedSurfaces, DetectedObjects


class TrayAndCupholderPerception(Node):
    """
    Detects:
      • Tray (as a plane/circular surface) → publishes /tray_marker (green cylinders) and /tray_detected
      • Cup-holders (as cylinders)         → publishes /cup_holder_marker (blue cylinders) and /cup_holder_detected
    """
    def __init__(self) -> None:
        super().__init__('tray_and_cupholder_detection_node')

        # Subscribe ONLY to the wrist depth cloud
        self.pc_sub_wrist = self.create_subscription(
            PointCloud2, '/wrist_rgbd_depth_sensor/points',
            self.callback_tray_and_cupholders, 10
        )

        # Publishers for markers and detections
        self.tray_marker_pub       = self.create_publisher(MarkerArray, '/tray_marker', 10)           # circular surfaces
        self.cupholder_marker_pub  = self.create_publisher(MarkerArray, '/cup_holder_marker', 10)

        self.tray_detected_pub     = self.create_publisher(DetectedSurfaces, '/tray_detected', 10)
        self.cuph_detected_pub     = self.create_publisher(DetectedObjects,  '/cup_holder_detected', 10)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    # ------------ Utilities ------------

    def from_ros_msg(self, msg: PointCloud2) -> Union[pcl.PointCloud, None]:
        """Converts a ROS2 PointCloud2 message to a PCL point cloud (in base_link)."""
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
            pts = []
            for i in range(n):
                s = i * step
                x = np.frombuffer(msg.data[s:s+4],       dtype=np.float32, count=1)[0]
                y = np.frombuffer(msg.data[s+4:s+8],     dtype=np.float32, count=1)[0]
                z = np.frombuffer(msg.data[s+8:s+12],    dtype=np.float32, count=1)[0]
                p_rel = R @ np.array([x, y, z], dtype=np.float32) + t
                pts.append(p_rel)

            arr = np.asarray(pts, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 3:
                return None
            cloud = pcl.PointCloud()
            cloud.from_array(arr)
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

    def filter_cloud(self, cloud: pcl.PointCloud,
                     min_x: float, max_x: float,
                     min_y: float, max_y: float,
                     min_z: float, max_z: float) -> Union[pcl.PointCloud, None]:
        """Axis-aligned ROI filter."""
        try:
            idx = []
            for i in range(cloud.size):
                px, py, pz = cloud[i]
                if (min_x <= px <= max_x and
                    min_y <= py <= max_y and
                    min_z <= pz <= max_z):
                    idx.append(i)
            return cloud.extract(idx)
        except Exception as e:
            self.get_logger().error(f"Error in filter_cloud: {e}")
            return None

    def extract_plane(self, cloud: pcl.PointCloud, use_normals: bool = True) -> Tuple[np.ndarray, np.ndarray, pcl.PointCloud]:
        """RANSAC plane extraction (with or without normals)."""
        seg = cloud.make_segmenter_normals(ksearch=50) if use_normals else cloud.make_segmenter()
        if use_normals:
            seg.set_optimize_coefficients(True)
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_distance_threshold(0.01)
        indices, coeffs = seg.segment()
        plane_cloud = cloud.extract(indices)
        return indices, coeffs, plane_cloud

    def extract_clusters(self, cloud: pcl.PointCloud, tol: float, min_sz: int, max_sz: int):
        """Euclidean clustering."""
        tree = cloud.make_kdtree()
        ec = cloud.make_EuclideanClusterExtraction()
        ec.set_ClusterTolerance(tol)
        ec.set_MinClusterSize(min_sz)
        ec.set_MaxClusterSize(max_sz)
        ec.set_SearchMethod(tree)
        cluster_indices = ec.Extract()

        clusters, centroids, dims = [], [], []
        for indices in cluster_indices:
            c = cloud.extract(indices)
            centroid = np.mean(c, axis=0)
            mn = np.min(c, axis=0)
            mx = np.max(c, axis=0)
            clusters.append(c)
            centroids.append(centroid.tolist())
            dims.append((mx - mn).tolist())
        return clusters, centroids, dims

    def extract_cylinder(self, cloud, max_cupholder=4, min_distance=0.05):
        """Extract cylindrical cup-holders."""
        cup_idx, cup_coeffs, cup_centroids = [], [], []
        work = cloud
        for _ in range(max_cupholder):
            seg = work.make_segmenter_normals(ksearch=50)
            seg.set_optimize_coefficients(True)
            seg.set_model_type(pcl.SACMODEL_CYLINDER)
            seg.set_normal_distance_weight(0.1)
            seg.set_method_type(pcl.SAC_RANSAC)
            seg.set_max_iterations(10000)
            seg.set_distance_threshold(0.05)
            seg.set_radius_limits(0.01, 0.06)
            indices, coefficients = seg.segment()
            if len(indices) < 30 or coefficients[6] > 0.06 or coefficients[6] < 0.01:
                break
            cup_idx.append(indices)
            cup_coeffs.append(coefficients)
            mask = np.ones(work.size, dtype=bool)
            mask[indices] = False
            work = work.extract(np.where(mask)[0])
            centroid = np.mean(cloud.extract(indices), axis=0)
            cup_centroids.append(centroid)

        filt_c, filt_i = [], []
        for i, c in enumerate(cup_centroids):
            if all(np.linalg.norm(np.array(c) - np.array(k)) >= min_distance for k in filt_c):
                filt_c.append(c); filt_i.append(cup_idx[i])

        merged = []
        for idxs in filt_i:
            merged.append(cloud.extract(idxs).to_array())
        if merged:
            merged = np.vstack(merged)
            cyl = pcl.PointCloud(); cyl.from_array(merged)
        else:
            cyl = pcl.PointCloud()
        return filt_i, cup_coeffs, cyl

    # ------------ Core callback (only wrist cloud) ------------

    def callback_tray_and_cupholders(self, msg: PointCloud2) -> None:
        try:
            cloud = self.from_ros_msg(msg)
            if cloud is None or cloud.size == 0:
                return

            # Tuned ROIs (adjust to your setup):
            # Tray region (plane-ish circular surface under holders)
            tray_roi = self.filter_cloud(cloud, -0.6, -0.2, -0.2, 0.2,  -0.65, -0.54)
            # Cup-holder region (where the cylinders stand)
            cuph_roi = self.filter_cloud(cloud, -0.55, -0.25, -0.15, 0.15, -0.63, -0.55)
            if tray_roi is None or cuph_roi is None:
                return

            # Extract tray plane
            _, _, tray_cloud = self.extract_plane(tray_roi, use_normals=True)
            # Extract cup-holder cylinders
            _, _, cuph_cloud = self.extract_cylinder(cuph_roi)

            # Cluster both to get centroids/dimensions
            _, tray_centroids, tray_dims = self.extract_clusters(tray_cloud, tol=0.04, min_sz=30, max_sz=100000)
            _, ch_centroids,   ch_dims   = self.extract_clusters(cuph_cloud, tol=0.04, min_sz=30, max_sz=100000)

            # Markers
            self.pub_circular_surface_markers(tray_centroids, tray_dims)
            self.pub_cup_holder_markers(ch_centroids, ch_dims)

            # Messages
            self.pub_surfaces_detected(self.tray_detected_pub, tray_centroids, tray_dims)
            self.pub_objects_detected(self.cuph_detected_pub, ch_centroids, ch_dims, fixed_height=0.035)

        except Exception as e:
            self.get_logger().error(f"[Tray/Cupholders] Error in callback: {e}")

    # ------------ Publishing helpers ------------

    def pub_circular_surface_markers(self, centroids, dims):
        """Publish tray (circular) surfaces as green cylinders."""
        ma = MarkerArray()
        for i, (c, d) in enumerate(zip(centroids, dims)):
            radius = float(d[0]) / 2.0
            height = 0.09
            m = Marker()
            m.header.frame_id = "base_link"
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y = float(c[0]), float(c[1])
            m.pose.position.z = float(c[2]) - height / 2.0
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = radius * 2.0
            m.scale.z = height
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.0, 0.4
            ma.markers.append(m)
        if ma.markers:
            self.tray_marker_pub.publish(ma)

    def pub_cup_holder_markers(self, centroids, dims):
        """Publish cup-holders as blue cylinders."""
        ma = MarkerArray()
        for i, (c, d) in enumerate(zip(centroids, dims)):
            radius = float(d[0]) / 2.0
            height = 0.035
            m = Marker()
            m.header.frame_id = "base_link"
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(c[0]), float(c[1]), float(c[2])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = radius * 2.0
            m.scale.z = height
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 0.0, 1.0, 0.9
            ma.markers.append(m)
        if ma.markers:
            self.cupholder_marker_pub.publish(ma)

    def pub_surfaces_detected(self, pub, centroids, dims):
        """Publish DetectedSurfaces (tray)."""
        if pub is None:
            return
        for i, (c, d) in enumerate(zip(centroids, dims)):
            m = DetectedSurfaces()
            m.surface_id = i
            m.position.x, m.position.y, m.position.z = float(c[0]), float(c[1]), float(c[2])
            m.height, m.width = float(d[0]), float(d[1])
            pub.publish(m)

    def pub_objects_detected(self, pub, centroids, dims, fixed_height: float = None):
        """Publish DetectedObjects (cup-holders)."""
        if pub is None:
            return
        for i, (c, d) in enumerate(zip(centroids, dims)):
            dia = float(max(d[0], d[1]))
            m = DetectedObjects()
            m.object_id = i
            m.position.x, m.position.y, m.position.z = float(c[0]), float(c[1]), float(c[2])
            m.width = dia
            m.thickness = dia
            m.height = float(d[2]) if fixed_height is None else fixed_height
            pub.publish(m)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrayAndCupholderPerception()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
