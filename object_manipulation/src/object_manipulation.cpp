#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

#include <moveit_msgs/msg/robot_trajectory.hpp>

#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>

#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

// program variables
static const rclcpp::Logger LOGGER = rclcpp::get_logger("move_group_node");
static const std::string PLANNING_GROUP_ROBOT = "ur_manipulator";
static const std::string PLANNING_GROUP_GRIPPER = "gripper";

class ObjectManipulation {
public:
  explicit ObjectManipulation(const rclcpp::Node::SharedPtr &base_node)
      : base_node_(base_node) {
    RCLCPP_INFO(LOGGER,
                "Initializing Class: Object Manipulation Trajectory...");

    // configure node options
    rclcpp::NodeOptions node_options;
    node_options.automatically_declare_parameters_from_overrides(true);

    // initialize move_group node
    move_group_node_ =
        rclcpp::Node::make_shared("move_group_node", node_options);

    // helper to set use_sim_time without double-declaring
    auto ensure_sim_time_true = [](const rclcpp::Node::SharedPtr &node) {
      try {
        if (!node->has_parameter("use_sim_time")) {
          node->declare_parameter<bool>("use_sim_time", true);
        }
      } catch (const rclcpp::exceptions::ParameterAlreadyDeclaredException &) {
        // already declared by overrides/launch; ignore
      }
      node->set_parameter(rclcpp::Parameter("use_sim_time", true));
    };

    // Make both nodes use sim time (without re-declaring)
    ensure_sim_time_true(move_group_node_);
    if (base_node_) {
      ensure_sim_time_true(base_node_);
    }

    // Wait until ROS time is active to avoid "stale joint state" warnings
    {
      RCLCPP_INFO(LOGGER, "[TIME] Waiting for ROS time to become active...");
      auto start = std::chrono::steady_clock::now();
      while (!move_group_node_->get_clock()->ros_time_is_active()) {
        if (std::chrono::steady_clock::now() - start >
            std::chrono::seconds(5)) {
          RCLCPP_WARN(LOGGER,
                      "[TIME] ROS time not active after 5s. Continuing.");
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
      RCLCPP_INFO(LOGGER, "[TIME] ros_time_is_active=%s",
                  move_group_node_->get_clock()->ros_time_is_active()
                      ? "true"
                      : "false");
    }

    // start move_group node in a new executor thread and spin it
    exec_.add_node(move_group_node_);
    std::thread([this]() { exec_.spin(); }).detach();

    // initialize move_group interfaces
    arm_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
        move_group_node_, PLANNING_GROUP_ROBOT);
    gripper_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
        move_group_node_, PLANNING_GROUP_GRIPPER);

    // Start state monitors so CurrentStateMonitor listens to /joint_states
    RCLCPP_INFO(LOGGER, "[INIT] Starting state monitors...");
    arm_->startStateMonitor();
    gripper_->startStateMonitor();

    // Wait up to 5s for a valid state (joint states to arrive)
    RCLCPP_INFO(LOGGER, "[INIT] Waiting for current state (up to 5s)...");
    {
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (std::chrono::steady_clock::now() < deadline) {
        auto st = arm_->getCurrentState(1); // wait up to 1s each try
        if (st)
          break;
        RCLCPP_WARN(LOGGER, "[INIT] Still waiting for /joint_states...");
      }
    }

    // print out basic system information
    RCLCPP_INFO(LOGGER, "Planning Frame: %s", arm_->getPlanningFrame().c_str());
    RCLCPP_INFO(LOGGER, "End Effector Link: %s",
                arm_->getEndEffectorLink().c_str());
    RCLCPP_INFO(LOGGER, "Available Planning Groups:");
    std::vector<std::string> group_names = arm_->getJointModelGroupNames();
    for (size_t i = 0; i < group_names.size(); i++) {
      RCLCPP_INFO(LOGGER, "Group %zu: %s", i, group_names[i].c_str());
    }

    // Some planning settings (like your perception-style script)
    arm_->setPlanningTime(5.0);
    // arm_->setNumPlanningAttempts(10);
    // arm_->setGoalPositionTolerance(0.005);
    // arm_->setGoalOrientationTolerance(0.05);
    arm_->setNumPlanningAttempts(20);
    arm_->setGoalPositionTolerance(0.1);
    arm_->setGoalOrientationTolerance(0.1);
    arm_->setMaxVelocityScalingFactor(0.3);
    arm_->setMaxAccelerationScalingFactor(0.3);

    arm_->setStartStateToCurrentState();
    gripper_->setStartStateToCurrentState();

    RCLCPP_INFO(LOGGER, "Class Initialized: Object Manipulation Trajectory");
  }

  ~ObjectManipulation() {
    RCLCPP_INFO(LOGGER, "Class Terminated: Object Manipulation Trajectory");
  }

  void execute_trajectory_plan() {
    RCLCPP_INFO(LOGGER,
                "Planning and Executing Object Manipulation Trajectory...");

    using Pose = geometry_msgs::msg::Pose;

    // --- Fixed cup pose (world frame) ---
    Pose cup_pose;
    cup_pose.position.x = 0.298;
    cup_pose.position.y = 0.330;
    cup_pose.position.z = 0.035;

    // 1) Go to named "home"
    RCLCPP_INFO(LOGGER, "Going to named pose: home ...");
    arm_->setNamedTarget("home");
    planAndExecArm("Step1_GoHome");
    arm_->setStartStateToCurrentState();
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // 2) Ensure gripper is "open" as the very first gripper action
    RCLCPP_INFO(LOGGER, "Opening Gripper (named pose: open)...");
    setGripperNamed("open");
    planAndExecGripper();
    gripper_->setStartStateToCurrentState();
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // 3) Move above the cup (Z-down orientation encoded in quaternion below)
    const double pregrasp_offset_z = 0.30; // 30cm above the cup pose
    RCLCPP_INFO(LOGGER,
                "Moving above cup pose (30cm) with Z-down orientation...");
    setGoalPoseTarget(cup_pose.position.x, cup_pose.position.y,
                      cup_pose.position.z + pregrasp_offset_z);
    planAndExecArm("Step3_PreGraspAboveCup");
    arm_->setStartStateToCurrentState();
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // 4) Approach straight down 12cm (Cartesian)
    RCLCPP_INFO(LOGGER, "Approaching (Cartesian down 12cm)...");
    cartesianDelta(+0.000, +0.000, -0.12, "Step4_ApproachDown");
    arm_->setStartStateToCurrentState();
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // 5) Close the gripper to grasp
    RCLCPP_INFO(LOGGER, "Closing Gripper...");
    setGripperNamed("close");
    planAndExecGripper();
    gripper_->setStartStateToCurrentState();
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // 6) Retreat straight up 30cm (Cartesian)
    RCLCPP_INFO(LOGGER, "Retreating (Cartesian up 30cm)...");
    cartesianDelta(+0.000, +0.000, +0.30, "Step6_RetreatUp");
    arm_->setStartStateToCurrentState();
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // 7) Rotate shoulder_pan_joint by +90°
    RCLCPP_INFO(LOGGER, "Rotating shoulder_pan_joint by +90 degrees...");
    rotateShoulderPan(+M_PI_2, "Step7_RotateShoulderPan_+90deg");
    // settle + resync to avoid "start deviates" on the next step
    std::this_thread::sleep_for(std::chrono::seconds(3));
    arm_->setStartStateToCurrentState();

    // 8) NEW: Move to the requested pose (Z-down orientation preserved)
    RCLCPP_INFO(LOGGER, "Moving to requested pose (x=-0.337211, y=-0.00417373, "
                        "z=-0.286098)...");
    setGoalPoseTarget(-0.337211f, -0.00417373f, -0.286098f);
    planAndExecArm("Step8_GoToRequestedPose");
    arm_->setStartStateToCurrentState();
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // 9) (original) Go to the previous final pose (optional; keep or remove)
    RCLCPP_INFO(LOGGER,
                "Moving to final cup pose (x=-0.337211, y=0.0, z=0.3)...");
    setGoalPoseTarget(-0.337211, 0.0, 0.3);
    planAndExecArm("Step9_GoToFinalPose");
    arm_->setStartStateToCurrentState();
    std::this_thread::sleep_for(std::chrono::seconds(3));

    RCLCPP_INFO(LOGGER, "Object Manipulation: finished sequence of steps.");
  }

private:
  using Plan = moveit::planning_interface::MoveGroupInterface::Plan;
  using RobotTrajectory = moveit_msgs::msg::RobotTrajectory;

  // Nodes/executor
  rclcpp::Node::SharedPtr base_node_;
  rclcpp::Node::SharedPtr move_group_node_;
  rclcpp::executors::SingleThreadedExecutor exec_;

  // MoveIt
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> gripper_;

  // ===== Helpers for arm =====
  void setGoalPoseTarget(float x, float y, float z) {
    geometry_msgs::msg::Pose target;
    target.position.x = x;
    target.position.y = y;
    target.position.z = z;
    // Z-down orientation encoded as a quaternion (-1,0,0,0) (180° about X)
    target.orientation.x = -1.0;
    target.orientation.y = 0.0;
    target.orientation.z = 0.0;
    target.orientation.w = 0.0;

    arm_->clearPoseTargets();
    arm_->setPoseTarget(target);
  }

  void planAndExecArm(const char *tag) {
    Plan plan;
    auto ok = (arm_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
    if (!ok) {
      RCLCPP_WARN(LOGGER, "[%s] arm plan failed", tag);
      return;
    }
    auto code = arm_->execute(plan);
    if (code != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(LOGGER, "[%s] arm execute failed (code=%d)", tag, code.val);
      return;
    }
    RCLCPP_INFO(LOGGER, "[%s] arm execute success", tag);
  }

  void cartesianDelta(double dx, double dy, double dz, const char *tag) {
    std::vector<geometry_msgs::msg::Pose> waypoints;
    auto cur = arm_->getCurrentPose().pose;
    waypoints.push_back(cur);
    cur.position.x += dx;
    cur.position.y += dy;
    cur.position.z += dz;
    waypoints.push_back(cur);

    RobotTrajectory traj;
    const double eef_step = 0.01;
    const double jump_thr = 0.0;
    double fraction =
        arm_->computeCartesianPath(waypoints, eef_step, jump_thr, traj);

    if (fraction <= 0.0) {
      RCLCPP_WARN(LOGGER, "[%s] cartesian plan failed (fraction=%.2f)", tag,
                  fraction);
      return;
    }

    auto code = arm_->execute(traj);
    if (code != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(LOGGER, "[%s] cartesian execute failed (code=%d)", tag,
                   code.val);
      return;
    }

    RCLCPP_INFO(LOGGER, "[%s] cartesian execute success (fraction=%.2f)", tag,
                fraction);
  }

  // Rotate shoulder_pan_joint by a relative angle (radians)
  void rotateShoulderPan(double delta_rad, const char *tag) {
    auto state = arm_->getCurrentState(1.0); // wait up to 1s
    if (!state) {
      RCLCPP_ERROR(LOGGER, "[%s] could not get current state", tag);
      return;
    }

    const moveit::core::JointModelGroup *jmg =
        state->getJointModelGroup(PLANNING_GROUP_ROBOT);
    if (!jmg) {
      RCLCPP_ERROR(LOGGER, "[%s] no JointModelGroup for %s", tag,
                   PLANNING_GROUP_ROBOT.c_str());
      return;
    }

    std::vector<double> joints;
    state->copyJointGroupPositions(jmg, joints);
    const auto &names = jmg->getVariableNames();

    // find shoulder_pan_joint index
    int idx = -1;
    for (size_t i = 0; i < names.size(); ++i) {
      if (names[i] == "shoulder_pan_joint") {
        idx = static_cast<int>(i);
        break;
      }
    }
    if (idx < 0 || static_cast<size_t>(idx) >= joints.size()) {
      RCLCPP_ERROR(LOGGER, "[%s] shoulder_pan_joint not found in group", tag);
      return;
    }

    // add delta and normalize to [-pi, pi]
    joints[idx] += delta_rad;
    while (joints[idx] > M_PI)
      joints[idx] -= 2.0 * M_PI;
    while (joints[idx] < -M_PI)
      joints[idx] += 2.0 * M_PI;

    arm_->setJointValueTarget(joints);

    Plan plan;
    auto ok = (arm_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
    if (!ok) {
      RCLCPP_WARN(LOGGER, "[%s] plan failed", tag);
      return;
    }
    auto code = arm_->execute(plan);
    if (code != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(LOGGER, "[%s] execute failed (code=%d)", tag, code.val);
      return;
    }
    RCLCPP_INFO(LOGGER, "[%s] execute success", tag);
  }

  // ===== Helpers for gripper =====
  void setGripperNamed(const std::string &name) {
    gripper_->setNamedTarget(name);
  }

  void planAndExecGripper() {
    Plan plan;
    auto ok = (gripper_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
    if (!ok) {
      RCLCPP_WARN(LOGGER, "gripper plan failed");
      return;
    }
    auto code = gripper_->execute(plan);
    if (code != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(LOGGER, "gripper execute failed (code=%d)", code.val);
      return;
    }
    RCLCPP_INFO(LOGGER, "gripper execute success");
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  auto base_node = std::make_shared<rclcpp::Node>("object_manipulation");
  ObjectManipulation app(base_node);
  app.execute_trajectory_plan();

  rclcpp::shutdown();
  return 0;
}