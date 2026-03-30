# object_detection package in object_detection2 folder

This folder contains a new modular implementation of the object_detection package.
The folder name is object_detection2 only to keep it separate in this repo.
The ROS package name inside is object_detection for drop-in replacement.

## Modules
- object_detection/perception_core.py: candidate extraction and scoring
- object_detection/geometry.py: depth sampling and camera projection
- object_detection/tracker.py: ID persistence and stability filtering
- object_detection/object_detection_node.py: ROS integration and publishers

## Behavior compatibility
- Node executable: object_detection
- QoS bridge executable: pcl_qos_conv
- Detection topic: /cup_holder_detected
- Marker topics: /cup_holder_markers and /cup_holder_marker
- Annotated image topics: /barista_cam_annotated and /tray_cam_annotated

## To use as replacement
Copy the contents of this folder into your real object_detection package folder in your ROS project.
