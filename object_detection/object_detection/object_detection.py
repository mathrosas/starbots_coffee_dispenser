#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, TransformStamped

import numpy as np
import pcl
import tf2_ros
from tf2_ros import TransformException, ConnectivityException
from typing import List, Tuple, Union

from custom_msgs.msg import DetectedSurfaces, DetectedObjects


# =========================
# Hard-coded configuration
# =========================

# Fixed cup pose (in base_link)
CUP_X = 0.298 #0.318
CUP_Y = 0.330 # 0.350
CUP_Z = 0.035 # as is
CUP_FRAME_PARENT = "base_link"
CUP_FRAME_CHILD  = "cup"

# Perception ROI (meters, in base_link)
ROI_MIN_X, ROI_MAX_X = -0.65, -0.15
ROI_MIN_Y, ROI_MAX_Y = -0.30,  0.30
ROI_MIN_Z, ROI_MAX_Z = -0.80,  0.10

# Below-tray band selection and radius cap (meters)
BELOW_BAND      = 0.06
Z_GAP_MIN       = 0.01
TRAY_RADIUS_CAP = 0.14

# Cup-holder physical gates (meters)
CUP_MIN_RADIUS = 0.028
CUP_MAX_RADIUS = 0.040
CUP_MIN_HEIGHT = 0.020
CUP_MAX_HEIGHT = 0.040

# Clustering parameters
CLUSTER_TOL = 0.04
CLUSTER_MIN = 30
CLUSTER_MAX = 100000

# Occupancy check
OCCUPANCY_Z_MARGIN   = 0.04
OCCUPANCY_PTS_THRESH = 20

# De-duplication
MIN_CENTROID_DISTANCE = 0.05

# Viz
TRAY_MARKER_HEIGHT = 0.09
CUP_MARKER_HEIGHT  = 0.035
TEXT_HEIGHT_OFFSET = 0.06

# Publish per-holder STATIC TFs
PUBLISH_HOLDER_TFS = True
# Use 'ch_' and 1-based numbering for names/frames (ch_1, ch_2, ...)
CUP_HOLDER_FRAME_PREFIX = "ch_"


class ObjectDetection(Node):
    """
    Detects:
      • Tray (as a plane-ish circular surface) → publishes /tray_marker (green cylinders) and /tray_detected
      • Cup-holders (as cylinders)             → publishes /cup_holder_marker (blue cylinders) and /cup_holder_detected

    Also:
      • Publishes a STATIC TF 'cup' at a hard-coded pose (CUP_X/Y/Z) in 'base_link'.
      • Publishes STATIC TFs 'ch_<n>' (1-based) for each detected holder, parented to 'base_link'.

    Option A:
      • Stop republishing holder TFs every callback.
      • Publish holder STATIC TFs once, and only re-publish if the set changes (count or centroid moves > eps).
    """

    def __init__(self) -> None:
        super().__init__('object_detection_node')

        # Runtime tuning
        self.declare_parameter('holder_offset_x', 0.0)
        self.declare_parameter('holder_offset_y', 0.0)
        self.declare_parameter('holder_offset_z', 0.0)
        self.declare_parameter('holder_match_max_dist', 0.08)
        self.declare_parameter('log_ordered_holders', False)

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

        # --- TF (listener + STATIC broadcasters) ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Static TF for fixed 'cup'
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        # Static TFs for cup holders (latched)
        self.holder_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        # --- Holder TF publish gating (Option A) ---
        self.holders_tf_published = False
        self.last_holder_centroids_raw: List[List[float]] = []
        self.last_holder_centroids: List[List[float]] = []
        self.holder_tf_eps = 0.005  # 5mm threshold; tweak if needed

        # Publish the static TF for the cup once at startup
        self.publish_static_cup_tf()
        self.get_logger().info(
            f"[object_detection] Published STATIC TF '{CUP_FRAME_CHILD}' in '{CUP_FRAME_PARENT}' "
            f"at ({CUP_X:.3f}, {CUP_Y:.3f}, {CUP_Z:.3f})"
        )

    # --------- Core callback ---------
    def callback_tray_and_cupholders(self, msg: PointCloud2) -> None:
        try:
            cloud = self.from_ros_msg(msg)
            if cloud is None or cloud.size == 0:
                return

            # 1) coarse ROI
            roi = self.filter_cloud(
                cloud,
                ROI_MIN_X, ROI_MAX_X,
                ROI_MIN_Y, ROI_MAX_Y,
                ROI_MIN_Z, ROI_MAX_Z
            )
            if roi is None or roi.size == 0:
                self.get_logger().warn("Empty ROI after coarse filter.")
                return

            # 2) segment tray plane from ROI, then cluster to get tray centroid/dims
            _, _, tray_plane = self.extract_plane(roi)
            tray_clusters, tray_centroids, tray_dims = self.extract_clusters(
                tray_plane, CLUSTER_TOL, CLUSTER_MIN, CLUSTER_MAX
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
                z_gap_min=Z_GAP_MIN,
                band_height=BELOW_BAND,
                radius_cap=TRAY_RADIUS_CAP
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
                min_r=CUP_MIN_RADIUS, max_r=CUP_MAX_RADIUS,
                min_h=CUP_MIN_HEIGHT, max_h=CUP_MAX_HEIGHT,
                tol=CLUSTER_TOL, min_sz=CLUSTER_MIN, max_sz=CLUSTER_MAX,
                occ_margin=OCCUPANCY_Z_MARGIN, occ_pts=OCCUPANCY_PTS_THRESH,
                min_dist=MIN_CENTROID_DISTANCE
            )

            # Keep IDs stable across callbacks (ch_1..ch_n) to prevent target swap.
            ch_centroids, ch_dims = self.stable_order_holders(
                ch_centroids, ch_dims, tray_center
            )

            if bool(self.get_parameter('log_ordered_holders').value) and ch_centroids:
                ordered = ", ".join(
                    [f"ch_{i+1}=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})"
                     for i, c in enumerate(ch_centroids)]
                )
                self.get_logger().info(f"[HOLDER_ORDER] {ordered}")

            # 5) publish markers + messages
            self.publish_tray([tray_center], [tray_dim_main])
            self.publish_cupholders(ch_centroids, ch_dims)
            self.last_holder_centroids_raw = [c[:] for c in ch_centroids]

        except Exception as e:
            self.get_logger().error(f"[ObjectDetection] Error in callback: {e}")

    # --------- Conversions / geometry ---------
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
            # zero-copy-ish parsing (assumes XYZ float32 at offsets 0/4/8)
            for i in range(n):
                s = i * step
                x = np.frombuffer(msg.data[s:s+4],    dtype=np.float32, count=1)[0]
                y = np.frombuffer(msg.data[s+4:s+8],  dtype=np.float32, count=1)[0]
                z = np.frombuffer(msg.data[s+8:s+12], dtype=np.float32, count=1)[0]
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

    # --------- Filtering / segmentation / clustering ---------
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
            z_thresh = cz + OCCUPANCY_Z_MARGIN
            r_thresh = radius + 0.01
            for i in range(reference_cloud.size):
                px, py, pz = reference_cloud[i]
                if pz > z_thresh and np.hypot(px - cx, py - cy) < r_thresh:
                    count += 1
                    if count >= OCCUPANCY_PTS_THRESH:
                        return True
            return False

        accepted_c = []
        accepted_d = []
        for c, d in zip(centroids, dims):
            arr_dim = np.asarray(d, dtype=float)
            diameter = float(max(arr_dim[0], arr_dim[1]))
            radius = diameter / 2.0
            height = float(arr_dim[2])

            if not (CUP_MIN_RADIUS <= radius <= CUP_MAX_RADIUS and
                    CUP_MIN_HEIGHT <= height <= CUP_MAX_HEIGHT):
                continue

            if looks_occupied(np.asarray(c, dtype=float), radius):
                self.get_logger().info("Skipping cluster (occupied by cup above).")
                continue

            too_close = any(np.linalg.norm(np.asarray(c) - np.asarray(k)) < MIN_CENTROID_DISTANCE
                            for k in accepted_c)
            if too_close:
                continue

            accepted_c.append(c)
            accepted_d.append([diameter, diameter, height])

        return accepted_c, accepted_d

    # --------- Holder TF gating helpers (Option A) ---------
    def _centroids_changed(self, centroids: List[List[float]], eps: float) -> bool:
        """True if count changed or any centroid moved more than eps."""
        if len(centroids) != len(self.last_holder_centroids):
            return True
        for a, b in zip(centroids, self.last_holder_centroids):
            if np.linalg.norm(np.asarray(a) - np.asarray(b)) > eps:
                return True
        return False

    def stable_order_holders(self,
                             centroids: List[List[float]],
                             dims: List[List[float]],
                             tray_center: List[float]) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Assign deterministic order to holders and keep it stable frame-to-frame.
        Initial order uses angle around tray center; subsequent frames match nearest
        previously ordered centroid within a distance gate.
        """
        if not centroids:
            return [], []

        tray_xy = np.asarray(tray_center[:2], dtype=float)
        pairs = list(zip(centroids, dims))
        pairs.sort(key=lambda p: float(np.arctan2(p[0][1] - tray_xy[1], p[0][0] - tray_xy[0])))

        if not self.last_holder_centroids_raw:
            ordered_c = [p[0] for p in pairs]
            ordered_d = [p[1] for p in pairs]
            return ordered_c, ordered_d

        prev = [np.asarray(c, dtype=float) for c in self.last_holder_centroids_raw]
        curr = [np.asarray(p[0], dtype=float) for p in pairs]
        max_match = float(self.get_parameter('holder_match_max_dist').value)

        used = set()
        ordered_pairs = []
        for p_prev in prev:
            best_i = -1
            best_d = float('inf')
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

        # Append unmatched detections in deterministic angle order
        for i, pair in enumerate(pairs):
            if i not in used:
                ordered_pairs.append(pair)

        ordered_c = [p[0] for p in ordered_pairs]
        ordered_d = [p[1] for p in ordered_pairs]
        return ordered_c, ordered_d

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
            m.scale.x = m.scale.y = radius * 2.0
            m.scale.z = h
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.0, 0.4
            ma.markers.append(m)

            msg = DetectedSurfaces()
            msg.surface_id = i
            msg.position.x, msg.position.y, msg.position.z = float(c[0]), float(c[1]), float(c[2])
            msg.height, msg.width = float(d[0]), float(d[1])
            self.tray_detected_pub.publish(msg)

        if ma.markers:
            self.tray_marker_pub.publish(ma)

    def publish_cupholders(self, centroids: List[List[float]], dims: List[List[float]]) -> None:
        off_x = float(self.get_parameter('holder_offset_x').value)
        off_y = float(self.get_parameter('holder_offset_y').value)
        off_z = float(self.get_parameter('holder_offset_z').value)

        shifted_centroids = [
            [float(c[0]) + off_x, float(c[1]) + off_y, float(c[2]) + off_z]
            for c in centroids
        ]

        ma = MarkerArray()
        h = float(CUP_MARKER_HEIGHT)
        text_h = float(TEXT_HEIGHT_OFFSET)

        for i, (c, d) in enumerate(zip(shifted_centroids, dims)):
            diameter = float(max(d[0], d[1]))
            radius = diameter / 2.0
            one_based = i + 1  # ch_1, ch_2, ...

            # Cylinder (blue)
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

            # Text label (BLUE)
            txt = Marker()
            txt.header.frame_id = "base_link"
            txt.id = i + 1000
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.text = f"ch_{one_based}"
            txt.pose.position.x = float(c[0])
            txt.pose.position.y = float(c[1])
            txt.pose.position.z = float(c[2]) + text_h
            txt.pose.orientation.w = 1.0
            txt.scale.x = txt.scale.y = txt.scale.z = 0.05
            txt.color.r = 0.0
            txt.color.g = 0.0
            txt.color.b = 1.0
            txt.color.a = 0.95
            ma.markers.append(txt)

            # Topic message (IDs 1-based to match names)
            m = DetectedObjects()
            m.object_id = one_based
            m.position = Point(x=float(c[0]), y=float(c[1]), z=float(c[2]))
            m.width = diameter
            m.thickness = diameter
            m.height = float(d[2])
            self.cuph_detected_pub.publish(m)

        self.cupholder_marker_pub.publish(ma)

        # Option A: Publish STATIC TFs once, and only update if the set changes
        if PUBLISH_HOLDER_TFS and shifted_centroids:
            if (not self.holders_tf_published) or self._centroids_changed(shifted_centroids, self.holder_tf_eps):
                self.publish_static_cupholder_tfs(shifted_centroids)
                self.holders_tf_published = True
                self.last_holder_centroids = [c[:] for c in shifted_centroids]
                self.get_logger().info("[TF] Cup-holder STATIC TFs updated (gated).")

    # --------- Static TF helpers ---------
    def publish_static_cup_tf(self) -> None:
        ts = TransformStamped()
        ts.header.stamp = self.get_clock().now().to_msg()
        ts.header.frame_id = CUP_FRAME_PARENT
        ts.child_frame_id  = CUP_FRAME_CHILD
        ts.transform.translation.x = CUP_X
        ts.transform.translation.y = CUP_Y
        ts.transform.translation.z = CUP_Z
        ts.transform.rotation.w = 1.0
        self.static_broadcaster.sendTransform(ts)

    def publish_static_cupholder_tfs(self, centroids: List[List[float]]) -> None:
        """
        Publish one STATIC TF per detected cup-holder:
            parent: base_link
            child : ch_<n>  (1-based)
        """
        if not centroids:
            return

        now = self.get_clock().now().to_msg()
        tfs: List[TransformStamped] = []
        for i, c in enumerate(centroids):
            one_based = i + 1
            ts = TransformStamped()
            ts.header.stamp = now
            ts.header.frame_id = CUP_FRAME_PARENT  # base_link
            ts.child_frame_id  = f"{CUP_HOLDER_FRAME_PREFIX}{one_based}"
            ts.transform.translation.x = float(c[0])
            ts.transform.translation.y = float(c[1])
            ts.transform.translation.z = float(c[2])
            ts.transform.rotation.w = 1.0  # identity orientation
            tfs.append(ts)

        self.holder_static_broadcaster.sendTransform(tfs)
        self.get_logger().info(f"Published {len(tfs)} STATIC TFs for cup-holders (ch_#).")


# ------------ main ------------
def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObjectDetection()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
