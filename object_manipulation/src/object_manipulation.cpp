#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

#include <moveit_msgs/msg/display_robot_state.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>

#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <thread>
#include <vector>

#include <tf2/LinearMath/Quaternion.h>

// program variables
static const rclcpp::Logger LOGGER = rclcpp::get_logger("move_group_node");
static const std::string PLANNING_GROUP_ROBOT = "ur_manipulator";
static const std::string PLANNING_GROUP_GRIPPER = "gripper";

class ObjectManipulation {
public:
  ObjectManipulation(rclcpp::Node::SharedPtr base_node_)
      : base_node_(base_node_) {
    RCLCPP_INFO(LOGGER,
                "Initializing Class: Object Manipulation Trajectory...");

    // configure node options
    rclcpp::NodeOptions node_options;
    // auto-declare parameters passed via overrides/launch
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
      // set it to true regardless
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
    executor_.add_node(move_group_node_);
    std::thread([this]() { this->executor_.spin(); }).detach();

    // initialize move_group interfaces
    move_group_robot_ = std::make_shared<MoveGroupInterface>(
        move_group_node_, PLANNING_GROUP_ROBOT);
    move_group_gripper_ = std::make_shared<MoveGroupInterface>(
        move_group_node_, PLANNING_GROUP_GRIPPER);

    // Start state monitors so CurrentStateMonitor listens to /joint_states
    RCLCPP_INFO(LOGGER, "[INIT] Starting state monitors...");
    move_group_robot_->startStateMonitor();
    move_group_gripper_->startStateMonitor();

    // Wait up to 5s for a valid state (joint states to arrive)
    RCLCPP_INFO(LOGGER, "[INIT] Waiting for current state (up to 5s)...");
    {
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (std::chrono::steady_clock::now() < deadline) {
        auto st =
            move_group_robot_->getCurrentState(1); // wait up to 1s each try
        if (st)
          break;
        RCLCPP_WARN(LOGGER, "[INIT] Still waiting for /joint_states...");
      }
    }

    // get initial state of robot and gripper
    auto st_robot = move_group_robot_->getCurrentState(1);
    auto st_grip = move_group_gripper_->getCurrentState(1);

    if (!st_robot) {
      RCLCPP_ERROR(LOGGER, "[INIT] No current robot state available. Aborting "
                           "init early to avoid segfaults.");
      return;
    }
    if (!st_grip) {
      RCLCPP_ERROR(LOGGER, "[INIT] No current gripper state available. Gripper "
                           "commands may fail.");
    }

    joint_model_group_robot_ =
        st_robot->getJointModelGroup(PLANNING_GROUP_ROBOT);
    joint_model_group_gripper_ =
        st_grip ? st_grip->getJointModelGroup(PLANNING_GROUP_GRIPPER) : nullptr;

    // print out basic system information
    RCLCPP_INFO(LOGGER, "Planning Frame: %s",
                move_group_robot_->getPlanningFrame().c_str());
    RCLCPP_INFO(LOGGER, "End Effector Link: %s",
                move_group_robot_->getEndEffectorLink().c_str());
    RCLCPP_INFO(LOGGER, "Available Planning Groups:");
    std::vector<std::string> group_names =
        move_group_robot_->getJointModelGroupNames();
    for (size_t i = 0; i < group_names.size(); i++) {
      RCLCPP_INFO(LOGGER, "Group %zu: %s", i, group_names[i].c_str());
    }

    // get current state of robot and gripper
    current_state_robot_ = move_group_robot_->getCurrentState(2);
    if (current_state_robot_ && joint_model_group_robot_) {
      current_state_robot_->copyJointGroupPositions(
          joint_model_group_robot_, joint_group_positions_robot_);
      RCLCPP_INFO(LOGGER, "[INIT] Robot joints vector size: %zu",
                  joint_group_positions_robot_.size());
    } else {
      RCLCPP_ERROR(
          LOGGER,
          "[INIT] current_state_robot_ or joint_model_group_robot_ is NULL");
    }

    current_state_gripper_ = move_group_gripper_->getCurrentState(2);
    if (current_state_gripper_ && joint_model_group_gripper_) {
      current_state_gripper_->copyJointGroupPositions(
          joint_model_group_gripper_, joint_group_positions_gripper_);
      RCLCPP_INFO(LOGGER, "[INIT] Gripper joints vector size: %zu",
                  joint_group_positions_gripper_.size());
    } else {
      RCLCPP_WARN(LOGGER, "[INIT] current_state_gripper_ or "
                          "joint_model_group_gripper_ is NULL");
    }

    // set start state of robot and gripper to current state
    move_group_robot_->setStartStateToCurrentState();
    move_group_gripper_->setStartStateToCurrentState();

    // indicate initialization
    RCLCPP_INFO(LOGGER, "Class Initialized: Object Manipulation Trajectory");
  }

  ~ObjectManipulation() {
    RCLCPP_INFO(LOGGER, "Class Terminated: Object Manipulation Trajectory");
  }

  void execute_trajectory_plan() {
    RCLCPP_INFO(LOGGER,
                "Planning and Executing Object Manipulation Trajectory...");

    // --- Fixed cup pose (world frame) ---
    Pose cup_pose;
    cup_pose.position.x = 0.298;
    cup_pose.position.y = 0.330;
    cup_pose.position.z = 0.035;

    // 1) Go to named "home" (no raw joint coordinates)
    RCLCPP_INFO(LOGGER, "Going to named pose: home ...");
    move_group_robot_->setNamedTarget("home");
    plan_trajectory_kinematics();
    execute_trajectory_kinematics();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // 2) Ensure gripper is "open" as the very first gripper action
    RCLCPP_INFO(LOGGER, "Opening Gripper (named pose: open)...");
    setup_named_pose_gripper("open");
    plan_trajectory_gripper();
    execute_trajectory_gripper();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // 3) Move above the cup with the TOOL POINTING DOWN (roll = pi)
    const double pregrasp_offset_z = 0.30; // 30cm above the cup pose
    tf2::Quaternion q_down;
    q_down.setRPY(M_PI, 0.0, 0.0); // roll=π → tool Z points down
    RCLCPP_INFO(LOGGER,
                "Moving above cup pose (30cm) with Z-down orientation...");
    setup_goal_pose_target(cup_pose.position.x, cup_pose.position.y,
                           cup_pose.position.z + pregrasp_offset_z);
    plan_trajectory_kinematics();
    execute_trajectory_kinematics();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // 4) Approach straight down 12cm (Cartesian)
    RCLCPP_INFO(LOGGER, "Approaching (Cartesian down 12cm)...");
    setup_waypoints_target(+0.000, +0.000, -0.12);
    plan_trajectory_cartesian();
    execute_trajectory_cartesian();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // 5) Close the gripper to grasp
    RCLCPP_INFO(LOGGER, "Closing Gripper...");
    setup_named_pose_gripper("close");
    plan_trajectory_gripper();
    execute_trajectory_gripper();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // 6) Retreat straight up 12cm (Cartesian)
    RCLCPP_INFO(LOGGER, "Retreating (Cartesian up 12cm)...");
    setup_waypoints_target(+0.000, +0.000, +0.12);
    plan_trajectory_cartesian();
    execute_trajectory_cartesian();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // 7) Go to the exact cup pose with gripper looking down
    RCLCPP_INFO(LOGGER, "Moving to final cup pose (x=-0.337211, y=-0.00417373, "
                        "z=-0.586098) with Z-down orientation...");
    setup_goal_pose_target(-0.337211, 0, 0);
    plan_trajectory_kinematics();
    execute_trajectory_kinematics();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // Done — no return-to-home here
    RCLCPP_INFO(LOGGER,
                "Object Manipulation: finished rotation left with cup.");
  }

private:
  using MoveGroupInterface = moveit::planning_interface::MoveGroupInterface;
  using JointModelGroup = moveit::core::JointModelGroup;
  using RobotStatePtr = moveit::core::RobotStatePtr;
  using Plan = MoveGroupInterface::Plan;
  using Pose = geometry_msgs::msg::Pose;
  using RobotTrajectory = moveit_msgs::msg::RobotTrajectory;

  rclcpp::Node::SharedPtr base_node_;
  rclcpp::Node::SharedPtr move_group_node_;
  rclcpp::executors::SingleThreadedExecutor executor_;

  std::shared_ptr<MoveGroupInterface> move_group_robot_;
  std::shared_ptr<MoveGroupInterface> move_group_gripper_;

  const JointModelGroup *joint_model_group_robot_{nullptr};
  const JointModelGroup *joint_model_group_gripper_{nullptr};

  std::vector<double> joint_group_positions_robot_;
  RobotStatePtr current_state_robot_;
  Plan kinematics_trajectory_plan_;
  Pose target_pose_robot_;
  bool plan_success_robot_ = false;

  std::vector<double> joint_group_positions_gripper_;
  RobotStatePtr current_state_gripper_;
  Plan gripper_trajectory_plan_;
  bool plan_success_gripper_ = false;

  std::vector<Pose> cartesian_waypoints_;
  RobotTrajectory cartesian_trajectory_plan_;
  const double jump_threshold_ = 0.0;
  const double end_effector_step_ = 0.01;
  double plan_fraction_robot_ = 0.0;

  void setup_joint_value_target(float a0, float a1, float a2, float a3,
                                float a4, float a5) {
    if (joint_group_positions_robot_.size() < 6) {
      RCLCPP_ERROR(
          LOGGER,
          "[setup_joint_value_target] robot joint vector size (%zu) < 6. "
          "Did CurrentStateMonitor receive joint states?",
          joint_group_positions_robot_.size());
      return;
    }
    joint_group_positions_robot_[0] = a0;
    joint_group_positions_robot_[1] = a1;
    joint_group_positions_robot_[2] = a2;
    joint_group_positions_robot_[3] = a3;
    joint_group_positions_robot_[4] = a4;
    joint_group_positions_robot_[5] = a5;
    move_group_robot_->setJointValueTarget(joint_group_positions_robot_);
  }

  void setup_goal_pose_target(float x, float y, float z) {
    target_pose_robot_.position.x = x;
    target_pose_robot_.position.y = y;
    target_pose_robot_.position.z = z;
    target_pose_robot_.orientation.x = -1;
    target_pose_robot_.orientation.y = 0;
    target_pose_robot_.orientation.z = 0;
    target_pose_robot_.orientation.w = 0;
    move_group_robot_->clearPoseTargets();
    move_group_robot_->setPoseTarget(target_pose_robot_);
  }

  void plan_trajectory_kinematics() {
    plan_success_robot_ =
        (move_group_robot_->plan(kinematics_trajectory_plan_) ==
         moveit::core::MoveItErrorCode::SUCCESS);
  }

  void execute_trajectory_kinematics() {
    if (plan_success_robot_) {
      auto code = move_group_robot_->execute(kinematics_trajectory_plan_);
      if (code != moveit::core::MoveItErrorCode::SUCCESS) {
        RCLCPP_ERROR(LOGGER, "Robot Kinematics Execute failed (code=%d)",
                     code.val);
      } else {
        RCLCPP_INFO(LOGGER, "Robot Kinematics Trajectory Success !");
      }
      move_group_robot_->setStartStateToCurrentState();
    } else {
      RCLCPP_INFO(LOGGER, "Robot Kinematics Trajectory Failed !");
    }
  }

  void setup_waypoints_target(float dx, float dy, float dz) {
    target_pose_robot_ = move_group_robot_->getCurrentPose().pose;
    cartesian_waypoints_.clear();
    cartesian_waypoints_.push_back(target_pose_robot_);
    target_pose_robot_.position.x += dx;
    target_pose_robot_.position.y += dy;
    target_pose_robot_.position.z += dz;
    cartesian_waypoints_.push_back(target_pose_robot_);
  }

  void plan_trajectory_cartesian() {
    if (cartesian_waypoints_.size() < 2) {
      RCLCPP_ERROR(LOGGER, "[cartesian] not enough waypoints (%zu)",
                   cartesian_waypoints_.size());
      plan_fraction_robot_ = -1.0;
      return;
    }
    plan_fraction_robot_ = move_group_robot_->computeCartesianPath(
        cartesian_waypoints_, end_effector_step_, jump_threshold_,
        cartesian_trajectory_plan_);
  }

  void execute_trajectory_cartesian() {
    if (plan_fraction_robot_ >= 0.0) {
      auto code = move_group_robot_->execute(cartesian_trajectory_plan_);
      if (code != moveit::core::MoveItErrorCode::SUCCESS) {
        RCLCPP_ERROR(LOGGER, "Robot Cartesian Execute failed (code=%d)",
                     code.val);
      } else {
        RCLCPP_INFO(LOGGER, "Robot Cartesian Trajectory Success !");
      }
      move_group_robot_->setStartStateToCurrentState();
    } else {
      RCLCPP_INFO(LOGGER, "Robot Cartesian Trajectory Failed !");
    }
    cartesian_waypoints_.clear();
  }

  void setup_joint_value_gripper(float angle) {
    if (joint_group_positions_gripper_.empty()) {
      RCLCPP_ERROR(LOGGER,
                   "[setup_joint_value_gripper] gripper joint vector is empty");
      return;
    }
    size_t idx = 2;
    if (idx >= joint_group_positions_gripper_.size()) {
      RCLCPP_WARN(LOGGER,
                  "[setup_joint_value_gripper] index 2 out of range "
                  "(size=%zu). Using last index.",
                  joint_group_positions_gripper_.size());
    }
    if (idx >= joint_group_positions_gripper_.size())
      idx = joint_group_positions_gripper_.size() - 1;
    joint_group_positions_gripper_[idx] = angle;
    move_group_gripper_->setJointValueTarget(joint_group_positions_gripper_);
  }

  void setup_named_pose_gripper(std::string pose_name) {
    move_group_gripper_->setNamedTarget(pose_name);
  }

  void plan_trajectory_gripper() {
    plan_success_gripper_ =
        (move_group_gripper_->plan(gripper_trajectory_plan_) ==
         moveit::core::MoveItErrorCode::SUCCESS);
  }

  void execute_trajectory_gripper() {
    if (plan_success_gripper_) {
      auto code = move_group_gripper_->execute(gripper_trajectory_plan_);
      if (code != moveit::core::MoveItErrorCode::SUCCESS) {
        RCLCPP_ERROR(LOGGER, "Gripper Execute failed (code=%d)", code.val);
      } else {
        RCLCPP_INFO(LOGGER, "Gripper Action Command Success !");
      }
      move_group_gripper_->setStartStateToCurrentState();
    } else {
      RCLCPP_INFO(LOGGER, "Gripper Action Command Failed !");
    }
  }
}; // class ObjectManipulation

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  auto base_node = std::make_shared<rclcpp::Node>("object_manipulation");

  ObjectManipulation app(base_node);
  app.execute_trajectory_plan();

  rclcpp::shutdown();
  return 0;
}
