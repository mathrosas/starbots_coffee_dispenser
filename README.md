# Starbots Coffee Cup Dispenser - Simulation

**Robotics Developer Masterclass - Final Project**

A ROS 2 Humble project that uses a simulated UR3e robotic arm with a Robotiq 85 gripper in Gazebo to autonomously detect cup holders on a barista robot, pick up coffee cups, and deliver them to a specified holder. The system integrates computer vision, motion planning, and a behavior tree-based task executor, running entirely in the Starbots cafeteria simulation environment.

![Simulation setup](./media/simulation_setup.png)

## Table of Contents

- [System Architecture](#system-architecture)
- [Packages Overview](#packages-overview)
  - [object\_detection](#object_detection---perception-pipeline)
  - [object\_manipulation](#object_manipulation---task-execution)
  - [my\_moveit\_config](#my_moveit_config---motion-planning)
  - [custom\_msgs](#custom_msgs---interface-definitions)
  - [universal\_robot\_ros2](#universal_robot_ros2---simulation-stack)
- [Perception System](#perception-system)
- [Manipulation System](#manipulation-system)
- [Web Interface (Foxglove)](#web-interface-foxglove)
- [Custom ROS 2 Interfaces](#custom-ros-2-interfaces)
- [Setup](#setup)
- [Usage](#usage)
- [Docker](#docker)
- [Troubleshooting](#troubleshooting)

## System Architecture

```
                        ┌──────────────────────┐
                        │   Wrist-Mounted      │
                        │   RGB-D Camera       │
                        │   (Gazebo Sensor)    │
                        └────┬─────┬─────┬─────┘
                             │     │     │
                           Color Depth PointCloud
                             │     │     │
                    ┌────────▼─────▼─────▼─────────┐
                    │   object_detection (Python)  │
                    │ ┌──────────────────────────┐ │
                    │ │ CupholderPerception      │ │
                    │ │ (contour + Hough circle) │ │
                    │ ├──────────────────────────┤ │
                    │ │ StableTracker            │ │
                    │ │ (EMA smoothing + NN)     │ │
                    │ └──────────────────────────┘ │
                    └─────────────┬────────────────┘
                                  │
                        /cup_holder_detected
                                  │
┌──────────────┐   ┌──────────────▼────────────────┐
│  Foxglove    │   │  object_manipulation (C++)    │
│  Web UI      │   │ ┌───────────────────────────┐ │
│  (optional)  │   │ │ BehaviorTree.CPP 4.6.2    │ │
│              │   │ │                           │ │
│ - 3D view    │   │ │ ValidateDetection         │ │
│ - Order btn  │◄──┤ │ PrePick ─► Pick           │ │
│ - Camera     │   │ │ PrePlace ─► Place         │ │
│ - Status     │   │ │ PutBack (recovery)        │ │
│ - Logs       │   │ │ Return (home)             │ │
└──────────────┘   │ └────────────┬──────────────┘ │
                   └──────────────┼────────────────┘
                                  │
                      ┌───────────▼────────────┐
                      │  MoveIt 2 Move Group   │
                      │ (OMPL + PILZ planners) │
                      └────────────┬───────────┘
                                   │
                      ┌────────────▼───────────┐
                      │  UR3e Gazebo Sim       │
                      │  (Starbots Cafeteria)  │
                      └────────────────────────┘
```

![RViz planning](./media/rviz_planning.png)

## Packages Overview

```
starbots_coffee_dispenser/
├── object_detection/        # Perception pipeline (Python)
├── object_manipulation/     # Task execution with BehaviorTree (C++)
├── my_moveit_config/        # MoveIt 2 config for UR3e + Robotiq 85
├── custom_msgs/             # ROS 2 message, service, and action definitions
├── universal_robot_ros2/    # UR3e simulation stack, Gazebo worlds, gripper models
└── media/                   # Screenshots for documentation
```

### `object_detection` - Perception Pipeline

An `ament_python` package that processes RGB-D camera data from the wrist-mounted Gazebo sensor to detect and track cup holders on the barista robot's tray.

**Nodes:**

| Node | Description |
|------|-------------|
| `object_detection` | Main perception node: detects cup holders, publishes 3D positions |

**Key modules:**

- `perception_core.py` - Cup holder detection using contour analysis with Hough Circle fallback
- `tracker.py` - Multi-object tracker with EMA smoothing and nearest-neighbor matching
- `geometry.py` - 3D point projection from depth images to robot-frame coordinates
- `object_detection.py` - Main ROS 2 node wiring perception, tracking, and TF transforms

### `object_manipulation` - Task Execution

A `CMake` C++ package that orchestrates the full pick-and-place pipeline using BehaviorTree.CPP 4.6.2 and MoveIt 2.

**Executables:**

| Executable | Description |
|------------|-------------|
| `object_manipulation` | BehaviorTree executor + MoveIt 2 motion interface |
| `deliver_cup_bridge` | Wraps the `/deliver_cup` action as a ROS 2 service for Foxglove integration |
| `add_cafeteria_scene` | Adds collision objects (tray, platform) to the MoveIt planning scene |

**BehaviorTree nodes** (8 total):

| Node | Type | Role |
|------|------|------|
| `GoalNotCanceled` | Condition | Checks if the delivery goal is still active |
| `ValidateDetection` | Action | Confirms the target cup holder has a valid detection |
| `PrePick` | Action | Moves to a pre-grasp pose above the cup |
| `Pick` | Action | Closes the gripper to grasp the cup |
| `PrePlace` | Action | Moves to a pre-place pose above the target holder |
| `Place` | Action | Opens the gripper to release the cup |
| `PutBack` | Action | Recovery: returns the cup to its original position |
| `Return` | Action | Returns the arm to the home configuration |

### `my_moveit_config` - Motion Planning

Generated with MoveIt Setup Assistant and customized for the UR3e + Robotiq 85 simulation setup.

**Planning groups:**

| Group | Description |
|-------|-------------|
| `ur_manipulator` | 6-DOF arm chain: `base_link` to `tool0` |
| `gripper` | Robotiq 85 parallel gripper |

**Named poses:** `home`, `flex`, `open`, `close`

**Planners:** OMPL (default: BiTRRT, sampling-based) and PILZ (deterministic industrial fallback)

**Trajectory resolution:** `longest_valid_segment_fraction: 0.005`, `maximum_waypoint_distance: 0.001`

### `custom_msgs` - Interface Definitions

Defines all project-specific ROS 2 interfaces used for inter-node communication.

### `universal_robot_ros2` - Simulation Stack

A collection of packages that provide the complete Gazebo simulation environment for the project.

**Key sub-packages:**

| Sub-package | Description |
|-------------|-------------|
| `Universal_Robots_ROS2_Description` | URDF/Xacro models for UR robot arms |
| `Universal_Robots_ROS2_Gazebo_Simulation` | Gazebo simulation plugins and configurations |
| `gazebo_ros2_control` | Bridge between Gazebo physics and ROS 2 control manager |
| `barista_ros2` | Barista robot description, delivery utilities, and table detection |
| `robotiq_85_gripper` | Robotiq 85 gripper model and simulation plugins |
| `the_construct_office_gazebo` | Complete Starbots cafeteria Gazebo world |

## Perception System

The perception pipeline runs as a single ROS 2 node that subscribes to the wrist-mounted RGB-D Gazebo sensor and publishes stable, filtered cup holder detections in the robot's `base_link` frame.

### Detection Algorithm

1. **Preprocessing** - CLAHE contrast enhancement + adaptive thresholding + morphological open/close
2. **Circle detection** - Primary: contour-based (circularity threshold 0.58). Fallback: Hough Circle transform
3. **3D projection** - Detected circles are projected to 3D using depth data and camera intrinsics
4. **Frame transform** - Positions are transformed from camera optical frame to `base_link` via TF2

### Multi-Object Tracking

Each detection is tracked across frames by a `StableTracker`:

- **Association:** Greedy nearest-neighbor matching (6cm threshold)
- **Smoothing:** Exponential Moving Average on position (alpha 0.30)
- **Confirmation:** A track must persist for 4 consecutive frames before it is published
- **Garbage collection:** Tracks with missed frames are removed after hold timeout (0.80s)

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/cup_holder_detected` | `custom_msgs/DetectedObjects` | Confirmed cup holder positions (3D, in `base_link`) |
| `/cup_holder_markers` | `visualization_msgs/MarkerArray` | RViz/Foxglove visualization markers |
| `/barista_cam_annotated` | `sensor_msgs/Image` | Annotated RGB image with detection overlays |

## Manipulation System

The manipulation node uses a BehaviorTree to sequence the pick-and-place pipeline. The tree is defined in `deliver_cup_tree.xml` and executed by BehaviorTree.CPP 4.6.2.

### Behavior Tree Structure

```
DeliverCupRoot (ReactiveSequence)
├── GoalNotCanceled                    ← checked on every tick
└── MainSequence
    ├── CoffeeDelivery
    │   ├── ValidateDetection
    │   ├── PrePick
    │   ├── Pick
    │   └── PlaceOrRecover (Fallback)
    │       ├── TrytoPlace
    │       │   ├── PrePlace
    │       │   └── Place
    │       ├── Inverter → PutBack     ← recovery: return cup
    │       └── Return                 ← last resort: go home
    └── Return
```

The `ReactiveSequence` at the root ensures the tree halts immediately if the goal is canceled. The `Fallback` node in `PlaceOrRecover` provides graceful degradation: if placing the cup fails, it attempts to put it back; if that also fails, it returns the arm home.

### Action Server

The main command interface is the `/deliver_cup` action, which accepts a target `cupholder_id` and reports progress through feedback messages. A service bridge (`deliver_cup_bridge`) also exposes the same functionality as a synchronous `/deliver_cup` service for use from the Foxglove web interface.

## Web Interface (Foxglove)

The project supports [Foxglove Studio](https://foxglove.dev/) as an optional web-based monitoring and control interface, connected via `rosbridge_server`.

When running through Docker (see [Docker](#docker)), the container automatically starts a `rosbridge_websocket` on port 9090. Connect Foxglove to:

- Same machine: `ws://localhost:9090`
- Remote server: `ws://<SERVER_IP>:9090`
- The Construct tunnel: use the URL returned by `rosbridge_address`

**Available panels:**

| Panel | Description |
|-------|-------------|
| **3D Visualization** | Robot model, cup holder markers, TF frames |
| **Order Coffee** | Service call button to trigger `/deliver_cup` with a target holder ID |
| **Robot Status** | Live feedback from `/robot_status_feedback` |
| **Barista Tray Detections** | Annotated camera stream from `/barista_cam_annotated` |
| **ROS Logs** | Filtered `/rosout` log viewer |

## Custom ROS 2 Interfaces

### Action: `DeliverCup`

Main command interface for triggering a cup delivery.

```
# Goal
uint32 cupholder_id          # Target cup holder (1-4)
---
# Result
bool success                 # True if cup delivered successfully
string message               # Status or error description
---
# Feedback
string stage                 # Current execution phase
float32 progress             # 0.0 to 1.0
uint32 cupholder_id          # Active target holder
```

### Message: `DetectedObjects`

Per-holder detection output from the perception node.

```
uint32 object_id                  # Unique tracked ID
geometry_msgs/Point position      # 3D position in base_link frame
float32 height                    # Holder dimension
float32 width                     # Holder dimension
float32 thickness                 # Rim thickness
```

### Message: `DetectedSurfaces`

Detected plane/surface (tray).

```
uint32 surface_id
geometry_msgs/Point position
float32 height
float32 width
```

### Service: `PickPlaceCup`

Synchronous wrapper used by the Foxglove service call panel.

```
# Request
uint8 goal_cup_holder
---
# Response
string result
```

## Setup

### Prerequisites

- Ubuntu 22.04 LTS
- ROS 2 Humble
- Python 3.10+
- Gazebo (Classic)

### 1. Install system dependencies

```bash
source /opt/ros/humble/setup.bash
sudo apt update
sudo apt install -y git cmake build-essential libzmq3-dev libsqlite3-dev python3-pcl
```

### 2. Build BehaviorTree.CPP 4.6.2

The manipulation package depends on BehaviorTree.CPP 4.6.2, which must be built from source:

```bash
git clone https://github.com/BehaviorTree/BehaviorTree.CPP.git
cd BehaviorTree.CPP
git checkout 4.6.2

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBTCPP_BUILD_TOOLS=ON \
  -DCMAKE_INSTALL_PREFIX=$HOME/.local

cmake --build build -j"$(nproc)"
cmake --install build
```

Verify:

```bash
find $HOME/.local -name 'behaviortree_cppConfig.cmake'
```

### 3. Initialize rosdep (first time only)

```bash
sudo rosdep init
rosdep update
```

### 4. Build the workspace

Clone this repository into your ROS 2 workspace `src/` directory, then:

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source ~/.local/share/behaviortree_cpp/local_setup.bash

colcon build --symlink-install \
  --cmake-args -Dbehaviortree_cpp_DIR=$HOME/.local/share/behaviortree_cpp/cmake

source install/setup.bash
```

## Usage

### 1. Launch the Gazebo simulation

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch the_construct_office_gazebo starbots_ur3e.launch.xml
```

> **Note:** The first launch may fail. Stop and relaunch — the second attempt should work.

### 2. Verify robot controllers

Confirm that the robot controllers are active:

```bash
ros2 control list_controllers
```

Expected:

```
joint_trajectory_controller[joint_trajectory_controller/JointTrajectoryController] active
joint_state_broadcaster[joint_state_broadcaster/JointStateBroadcaster] active
gripper_controller  [position_controllers/GripperActionController] active
```

Confirm joint states are streaming:

```bash
ros2 topic echo /joint_states --once
```

### 3. Verify camera topics

```bash
ros2 topic list | grep -E "depth|camera|point"
```

### 4. Launch the manipulation pipeline

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch object_manipulation deliver_cup.launch.py
```

This single launch file starts all required nodes:

- Object detection (perception)
- MoveIt Move Group
- Collision scene setup
- RViz visualization
- BehaviorTree manipulation node (delayed start)
- Deliver cup service bridge

### 5. Send a delivery command

In another terminal:

```bash
source ~/ros2_ws/install/setup.bash
ros2 action send_goal /deliver_cup custom_msgs/action/DeliverCup "{cupholder_id: 1}" --feedback
```

Or use the Foxglove web interface to press **Order Coffee** with the desired holder ID.

## Docker

This repository includes a dockerized runtime adapted to this project.

From the project root:

```bash
cd ~/ros2_ws/src/starbots_coffee_dispenser/docker
chmod +x ros_entrypoint.sh
docker compose build
docker compose up
```

What this container launches:

- `ros2 launch object_manipulation deliver_cup.launch.py`
- `ros2 launch rosbridge_server rosbridge_websocket_launch.xml port:=9090`

**Important:** This container expects the Gazebo simulation to already be running in the same ROS domain. Start the simulation on the host (outside the container) before sending goals:

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch the_construct_office_gazebo starbots_ur3e.launch.xml
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `/deliver_cup` does nothing | Verify controllers are `active` with `ros2 control list_controllers`, check `/joint_states` is publishing, relaunch the manipulation launch file |
| No camera topics | Ensure the Gazebo simulation has fully spawned, then check depth/camera topics with `ros2 topic list` |
| Detection not publishing | Verify camera sensor data is streaming with `ros2 topic list` |
| First Gazebo launch fails | Relaunch once — this is a known intermittent behavior |
| Simulation behaves unstable | Wait until Gazebo fully spawns both robots before sending action goals |
