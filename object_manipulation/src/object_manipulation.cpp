#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

#include <custom_msgs/msg/detected_objects.hpp>
#include <moveit_msgs/msg/display_robot_state.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>

#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>

#include <chrono>
#include <cmath>
#include <functional>
#include <future>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// program variables
static const rclcpp::Logger LOGGER = rclcpp::get_logger("move_group_node");
static const std::string PLANNING_GROUP_ROBOT = "ur_manipulator";
static const std::string PLANNING_GROUP_GRIPPER = "gripper";
static const std::string REF_FRAME = "base_link";
static const std::string CUPHOLDER_TOPIC = "/cup_holder_detected";

// offsets / “magic numbers”:
static constexpr double PREGRASP_Z_OFFSET = 0.30; // 20 cm above detected object
static constexpr double APPROACH_Z_DELTA = -0.12; // straight down 8.5 cm
static constexpr double RETREAT_Z_DELTA = +0.30;  // straight up 8.5 cm

// project defaults for fixed-cup mode
static constexpr double FIXED_CUP_X = 0.299;
static constexpr double FIXED_CUP_Y = 0.331;
static constexpr double FIXED_CUP_Z = 0.035;

class PickAndPlacePerception {
public:
  using DetectedObject = custom_msgs::msg::DetectedObjects;

  PickAndPlacePerception(rclcpp::Node::SharedPtr base_node_)
      : base_node_(base_node_) {
    RCLCPP_INFO(LOGGER, "Initializing Class: Pick And Place Perception...");

    rclcpp::NodeOptions node_options;
    node_options.automatically_declare_parameters_from_overrides(true);

    // initialize move_group node
    move_group_node_ =
        rclcpp::Node::make_shared("move_group_node", node_options);
    ensure_sim_time_true(move_group_node_);
    if (base_node_) {
      ensure_sim_time_true(base_node_);
    }

    if (!move_group_node_->has_parameter("cup_x")) {
      move_group_node_->declare_parameter<double>("cup_x", FIXED_CUP_X);
    }
    if (!move_group_node_->has_parameter("cup_y")) {
      move_group_node_->declare_parameter<double>("cup_y", FIXED_CUP_Y);
    }
    if (!move_group_node_->has_parameter("cup_z")) {
      move_group_node_->declare_parameter<double>("cup_z", FIXED_CUP_Z);
    }
    if (!move_group_node_->has_parameter("ch")) {
      move_group_node_->declare_parameter<std::string>("ch", "ch_1");
    }
    const auto ch = move_group_node_->get_parameter("ch").as_string();
    target_holder_id_ = parse_ch_to_id(ch);

    executor_.add_node(move_group_node_);
    executor_thread_ = std::thread([this]() { this->executor_.spin(); });

    // initialize move_group interfaces
    move_group_robot_ = std::make_shared<MoveGroupInterface>(
        move_group_node_, PLANNING_GROUP_ROBOT);
    move_group_gripper_ = std::make_shared<MoveGroupInterface>(
        move_group_node_, PLANNING_GROUP_GRIPPER);

    move_group_robot_->setPoseReferenceFrame(REF_FRAME);
    move_group_robot_->setPlanningTime(8.0);
    move_group_robot_->setNumPlanningAttempts(20);
    move_group_robot_->setGoalPositionTolerance(0.001);
    move_group_robot_->setGoalOrientationTolerance(0.05);
    move_group_robot_->setMaxVelocityScalingFactor(0.3);
    move_group_robot_->setMaxAccelerationScalingFactor(0.3);

    move_group_gripper_->setGoalTolerance(0.0001);
    move_group_gripper_->setMaxVelocityScalingFactor(
        0.01); // Slow for precise/less jittery close
    move_group_gripper_->setMaxAccelerationScalingFactor(0.01);

    move_group_robot_->startStateMonitor();
    move_group_gripper_->startStateMonitor();

    current_state_robot_ = move_group_robot_->getCurrentState(10);
    if (!current_state_robot_) {
      throw std::runtime_error(
          "Could not get current robot state for arm group.");
    }
    current_state_gripper_ = move_group_gripper_->getCurrentState(10);
    if (!current_state_gripper_) {
      throw std::runtime_error(
          "Could not get current robot state for gripper group.");
    }

    joint_model_group_robot_ =
        current_state_robot_->getJointModelGroup(PLANNING_GROUP_ROBOT);
    joint_model_group_gripper_ =
        current_state_gripper_->getJointModelGroup(PLANNING_GROUP_GRIPPER);

    cupholder_promise_ = std::make_shared<std::promise<DetectedObject>>();
    cupholder_future_ = cupholder_promise_->get_future();
    cupholder_sub_ = move_group_node_->create_subscription<DetectedObject>(
        CUPHOLDER_TOPIC, 10,
        std::bind(&PickAndPlacePerception::cupholder_callback, this,
                  std::placeholders::_1));

    // print out basic system information
    RCLCPP_INFO(LOGGER, "Planning Frame: %s",
                move_group_robot_->getPlanningFrame().c_str());
    RCLCPP_INFO(LOGGER, "End Effector Link: %s",
                move_group_robot_->getEndEffectorLink().c_str());
    RCLCPP_INFO(LOGGER, "Target cupholder param ch='%s' -> id=%d", ch.c_str(),
                target_holder_id_);

    for (const auto &name : move_group_robot_->getJointModelGroupNames()) {
      RCLCPP_INFO(LOGGER, "Available group: %s", name.c_str());
    }

    // get current state of robot and gripper
    current_state_robot_->copyJointGroupPositions(joint_model_group_robot_,
                                                  joint_group_positions_robot_);
    current_state_gripper_->copyJointGroupPositions(
        joint_model_group_gripper_, joint_group_positions_gripper_);

    // set start state of robot and gripper to current state
    move_group_robot_->setStartStateToCurrentState();
    move_group_gripper_->setStartStateToCurrentState();

    // indicate initialization
    RCLCPP_INFO(LOGGER, "Class Initialized: Pick And Place Perception");
  }

  ~PickAndPlacePerception() {
    executor_.cancel();
    if (executor_thread_.joinable()) {
      executor_thread_.join();
    }
    RCLCPP_INFO(LOGGER, "Class Terminated: Pick And Place Perception");
  }

  void execute_trajectory_plan() {
    using namespace std::chrono_literals;

    RCLCPP_INFO(LOGGER, "Waiting for target cupholder id=%d on %s ...",
                target_holder_id_, CUPHOLDER_TOPIC.c_str());
    while (rclcpp::ok()) {
      if (cupholder_future_.wait_for(5s) == std::future_status::ready) {
        break;
      }
      RCLCPP_INFO(LOGGER, "No target cupholder yet, waiting another 5s...");
    }
    if (!rclcpp::ok()) {
      return;
    }
    const auto holder = cupholder_future_.get();
    RCLCPP_INFO(LOGGER,
                "Using cupholder id=%u pose=(%.3f, %.3f, %.3f) w=%.3f h=%.3f "
                "t=%.3f",
                holder.object_id, holder.position.x, holder.position.y,
                holder.position.z, holder.width, holder.height,
                holder.thickness);

    const double cup_x = move_group_node_->get_parameter("cup_x").as_double();
    const double cup_y = move_group_node_->get_parameter("cup_y").as_double();
    const double cup_z = move_group_node_->get_parameter("cup_z").as_double();

    const double pre_x = cup_x;
    const double pre_y = cup_y;
    const double pre_z = cup_z + PREGRASP_Z_OFFSET;
    const double place_x = holder.position.x;
    const double place_y = holder.position.y;
    const double place_z = holder.position.z;

    RCLCPP_INFO(LOGGER, "Using fixed cup pose at (%.3f, %.3f, %.3f)", cup_x,
                cup_y, cup_z);
    RCLCPP_INFO(LOGGER, "Planning and Executing Pick And Place Perception...");

    // 1. go to pregrasp
    RCLCPP_INFO(LOGGER, "Going to Pregrasp Position (%.3f, %.3f, %.3f)...",
                pre_x, pre_y, pre_z);
    setup_goal_pose_target(pre_x, pre_y, pre_z, -1.000, +0.000, +0.000, +0.000);
    plan_trajectory_kinematics();
    if (!execute_trajectory_kinematics()) {
      return;
    }

    // wait for few seconds
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // 2. open the gripper
    RCLCPP_INFO(LOGGER, "Opening Gripper...");
    setup_named_pose_gripper("open");
    plan_trajectory_gripper();
    if (!execute_trajectory_gripper()) {
      return;
    }

    // wait for few seconds
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // 3. approach straight down
    RCLCPP_INFO(LOGGER, "Approaching object...");
    setup_waypoints_target(+0.000, +0.000, APPROACH_Z_DELTA);
    plan_trajectory_cartesian();
    if (!execute_trajectory_cartesian()) {
      return;
    }

    // wait for few seconds
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // 4. close the gripper
    RCLCPP_INFO(LOGGER, "Closing Gripper...");
    setup_named_pose_gripper("close");
    plan_trajectory_gripper();
    if (!execute_trajectory_gripper()) {
      return;
    }

    // wait for few seconds
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // 5. retreat
    RCLCPP_INFO(LOGGER, "Retreating...");
    setup_waypoints_target(+0.000, +0.000, RETREAT_Z_DELTA);
    plan_trajectory_cartesian();
    if (!execute_trajectory_cartesian()) {
      return;
    }

    // wait for few seconds
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // 6. rotate shoulder
    RCLCPP_INFO(LOGGER, "Rotate Shoulder Joint 120 degrees...");
    current_state_robot_ = move_group_robot_->getCurrentState(10);
    current_state_robot_->copyJointGroupPositions(joint_model_group_robot_,
                                                  joint_group_positions_robot_);
    setup_joint_value_target(
        joint_group_positions_robot_[0] + 2.0 * M_PI / 3.0, // add 120 degrees
        joint_group_positions_robot_[1], joint_group_positions_robot_[2],
        joint_group_positions_robot_[3], joint_group_positions_robot_[4],
        joint_group_positions_robot_[5]);
    plan_trajectory_kinematics();
    if (!execute_trajectory_kinematics()) {
      return;
    }

    // wait for few seconds
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // 7. go to pre-place position (from detected cupholder)
    RCLCPP_INFO(LOGGER, "Going to Pre-place Position (%.3f, %.3f, %.3f)...",
                place_x, place_y, place_z + 2*PREGRASP_Z_OFFSET);
    setup_goal_pose_target(place_x, place_y,
                           place_z + 2*PREGRASP_Z_OFFSET, -1.000,
                           +0.000, +0.000, +0.000);
    plan_trajectory_kinematics();
    if (!execute_trajectory_kinematics()) {
      return;
    }

    // wait for few seconds
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // 8. go to lower pre-place position (from detected cupholder)
    RCLCPP_INFO(LOGGER, "Going to Lower Pre-place Position (%.3f, %.3f, %.3f)...",
                place_x, place_y, place_z + PREGRASP_Z_OFFSET - 0.02);
    setup_goal_pose_target(place_x, place_y,
                           place_z + PREGRASP_Z_OFFSET - 0.02, -1.000,
                           +0.000, +0.000, +0.000);
    plan_trajectory_kinematics();
    if (!execute_trajectory_kinematics()) {
      return;
    }

    // wait for few seconds
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // 9. approach down to cupholder center
    RCLCPP_INFO(LOGGER, "Approaching down to Place Position (%.3f, %.3f, %.3f)...",
                place_x, place_y, place_z);
    setup_waypoints_target(+0.000, +0.000, -0.10);
    plan_trajectory_cartesian();
    if (!execute_trajectory_cartesian()) {
      return;
    }

    // wait for few seconds
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // 10. open the gripper
    RCLCPP_INFO(LOGGER, "Opening Gripper...");
    setup_named_pose_gripper("open");
    plan_trajectory_gripper();
    if (!execute_trajectory_gripper()) {
      return;
    }

    // wait for few seconds
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // 11. Going to Initial Position
    RCLCPP_INFO(LOGGER, "Going to Initial Position...");
    setup_joint_value_target(+0.0000, -1.5708, +0.0000, -1.5708, +0.0000,
                             +0.0000);
    plan_trajectory_kinematics();
    if (!execute_trajectory_kinematics()) {
      return;
    }

    RCLCPP_INFO(LOGGER, "Pick And Place Perception Execution Complete");
  }

private:
  // using shorthand for lengthy class references
  using MoveGroupInterface = moveit::planning_interface::MoveGroupInterface;
  using JointModelGroup = moveit::core::JointModelGroup;
  using RobotStatePtr = moveit::core::RobotStatePtr;
  using Plan = MoveGroupInterface::Plan;
  using Pose = geometry_msgs::msg::Pose;
  using RobotTrajectory = moveit_msgs::msg::RobotTrajectory;

  rclcpp::Node::SharedPtr base_node_;
  rclcpp::Node::SharedPtr move_group_node_;
  rclcpp::executors::SingleThreadedExecutor executor_;
  std::thread executor_thread_;

  std::shared_ptr<MoveGroupInterface> move_group_robot_;
  std::shared_ptr<MoveGroupInterface> move_group_gripper_;

  const JointModelGroup *joint_model_group_robot_{nullptr};
  const JointModelGroup *joint_model_group_gripper_{nullptr};

  std::vector<double> joint_group_positions_robot_;
  RobotStatePtr current_state_robot_;
  Plan kinematics_trajectory_plan_;
  Pose target_pose_robot_;
  bool plan_success_robot_{false};

  std::vector<double> joint_group_positions_gripper_;
  RobotStatePtr current_state_gripper_;
  Plan gripper_trajectory_plan_;
  bool plan_success_gripper_{false};

  std::vector<Pose> cartesian_waypoints_;
  RobotTrajectory cartesian_trajectory_plan_;
  const double jump_threshold_{0.0};
  const double end_effector_step_{0.01};
  double plan_fraction_robot_{0.0};

  int target_holder_id_{1};
  std::shared_ptr<std::promise<DetectedObject>> cupholder_promise_;
  std::future<DetectedObject> cupholder_future_;
  rclcpp::Subscription<DetectedObject>::SharedPtr cupholder_sub_;
  bool cupholder_received_{false};

  static void ensure_sim_time_true(const rclcpp::Node::SharedPtr &node) {
    try {
      if (!node->has_parameter("use_sim_time")) {
        node->declare_parameter<bool>("use_sim_time", true);
      }
    } catch (const rclcpp::exceptions::ParameterAlreadyDeclaredException &) {
    }
    node->set_parameter(rclcpp::Parameter("use_sim_time", true));
  }

  static int parse_ch_to_id(const std::string &s) {
    try {
      if (s.rfind("ch_", 0) == 0) {
        const int id = std::stoi(s.substr(3));
        return (id >= 1) ? id : 1;
      }
      const int id = std::stoi(s);
      return (id >= 1) ? id : 1;
    } catch (...) {
      return 1;
    }
  }

  void cupholder_callback(const DetectedObject::SharedPtr msg) {
    if (cupholder_received_) {
      return;
    }
    if (static_cast<int>(msg->object_id) != target_holder_id_) {
      return;
    }
    cupholder_received_ = true;
    cupholder_promise_->set_value(*msg);
    cupholder_sub_.reset();
    RCLCPP_INFO(LOGGER, "Received target cupholder id=%u at (%.3f, %.3f, %.3f)",
                msg->object_id, msg->position.x, msg->position.y,
                msg->position.z);
  }

  void setup_joint_value_target(float angle0, float angle1, float angle2,
                                float angle3, float angle4, float angle5) {
    joint_group_positions_robot_.resize(6);
    joint_group_positions_robot_[0] = angle0; // Shoulder Pan
    joint_group_positions_robot_[1] = angle1; // Shoulder Lift
    joint_group_positions_robot_[2] = angle2; // Elbow
    joint_group_positions_robot_[3] = angle3; // Wrist 1
    joint_group_positions_robot_[4] = angle4; // Wrist 2
    joint_group_positions_robot_[5] = angle5; // Wrist 3
    move_group_robot_->setStartStateToCurrentState();
    move_group_robot_->setJointValueTarget(joint_group_positions_robot_);
  }

  void setup_goal_pose_target(float pos_x, float pos_y, float pos_z,
                              float quat_x, float quat_y, float quat_z,
                              float quat_w) {
    target_pose_robot_.position.x = pos_x;
    target_pose_robot_.position.y = pos_y;
    target_pose_robot_.position.z = pos_z;
    target_pose_robot_.orientation.x = quat_x;
    target_pose_robot_.orientation.y = quat_y;
    target_pose_robot_.orientation.z = quat_z;
    target_pose_robot_.orientation.w = quat_w;
    move_group_robot_->setStartStateToCurrentState();
    move_group_robot_->setPoseTarget(target_pose_robot_);
  }

  void plan_trajectory_kinematics() {
    plan_success_robot_ =
        (move_group_robot_->plan(kinematics_trajectory_plan_) ==
         moveit::core::MoveItErrorCode::SUCCESS);
  }

  bool execute_trajectory_kinematics() {
    if (plan_success_robot_) {
      log_ee_and_joints("pre_execute_kinematics");
      auto code = move_group_robot_->execute(kinematics_trajectory_plan_);
      if (code != moveit::core::MoveItErrorCode::SUCCESS) {
        RCLCPP_ERROR(LOGGER, "Robot Kinematics Trajectory Execute Failed !");
        return false;
      }
      RCLCPP_INFO(LOGGER, "Robot Kinematics Trajectory Success !");
      log_ee_and_joints("post_execute_kinematics");
      return true;
    }
    RCLCPP_ERROR(LOGGER, "Robot Kinematics Trajectory Planning Failed !");
    return false;
  }

  void setup_waypoints_target(float x_delta, float y_delta, float z_delta) {
    cartesian_waypoints_.clear();
    target_pose_robot_ = move_group_robot_->getCurrentPose().pose;
    cartesian_waypoints_.push_back(target_pose_robot_);
    target_pose_robot_.position.x += x_delta;
    target_pose_robot_.position.y += y_delta;
    target_pose_robot_.position.z += z_delta;
    cartesian_waypoints_.push_back(target_pose_robot_);
  }

  void plan_trajectory_cartesian() {
    plan_fraction_robot_ = move_group_robot_->computeCartesianPath(
        cartesian_waypoints_, end_effector_step_, jump_threshold_,
        cartesian_trajectory_plan_);
  }

  bool execute_trajectory_cartesian() {
    // keep the original template behavior: execute for any non-negative
    // fraction
    if (plan_fraction_robot_ >= 0.0) {
      log_ee_and_joints("pre_execute_cartesian");
      auto code = move_group_robot_->execute(cartesian_trajectory_plan_);
      if (code != moveit::core::MoveItErrorCode::SUCCESS) {
        RCLCPP_ERROR(LOGGER, "Robot Cartesian Trajectory Execute Failed !");
        return false;
      }
      RCLCPP_INFO(LOGGER,
                  "Robot Cartesian Trajectory Success ! (fraction=%.2f)",
                  plan_fraction_robot_);
      log_ee_and_joints("post_execute_cartesian");
      return true;
    }
    RCLCPP_ERROR(LOGGER, "Robot Cartesian Trajectory Failed !");
    return false;
  }

  void setup_named_pose_gripper(std::string pose_name) {
    move_group_gripper_->setStartStateToCurrentState();
    move_group_gripper_->setNamedTarget(pose_name);
  }

  void plan_trajectory_gripper() {
    plan_success_gripper_ =
        (move_group_gripper_->plan(gripper_trajectory_plan_) ==
         moveit::core::MoveItErrorCode::SUCCESS);
  }

  bool execute_trajectory_gripper() {
    if (plan_success_gripper_) {
      auto code = move_group_gripper_->execute(gripper_trajectory_plan_);
      if (code != moveit::core::MoveItErrorCode::SUCCESS) {
        RCLCPP_ERROR(LOGGER, "Gripper Action Command Execute Failed !");
        return false;
      }
      RCLCPP_INFO(LOGGER, "Gripper Action Command Success !");
      return true;
    }
    RCLCPP_ERROR(LOGGER, "Gripper Action Command Planning Failed !");
    return false;
  }

  void log_ee_and_joints(const std::string &tag) {
    const auto ee = move_group_robot_->getCurrentPose().pose;
    auto state = move_group_robot_->getCurrentState(2.0);
    if (!state || !joint_model_group_robot_) {
      RCLCPP_WARN(LOGGER,
                  "[STATE:%s] Unable to read current robot state for joints.",
                  tag.c_str());
      return;
    }

    std::vector<double> joints;
    state->copyJointGroupPositions(joint_model_group_robot_, joints);

    std::ostringstream joint_ss;
    joint_ss << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < joints.size(); ++i) {
      if (i > 0) {
        joint_ss << ", ";
      }
      joint_ss << "j" << (i + 1) << "=" << joints[i];
    }

    RCLCPP_INFO(LOGGER,
                "[STATE:%s] ee=(%.3f, %.3f, %.3f) quat=(%.3f, %.3f, %.3f, "
                "%.3f) joints=[%s]",
                tag.c_str(), ee.position.x, ee.position.y, ee.position.z,
                ee.orientation.x, ee.orientation.y, ee.orientation.z,
                ee.orientation.w, joint_ss.str().c_str());
  }
}; // class PickAndPlacePerception

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  std::shared_ptr<rclcpp::Node> base_node =
      std::make_shared<rclcpp::Node>("pick_and_place_perception");
  PickAndPlacePerception pick_and_place(base_node);
  pick_and_place.execute_trajectory_plan();
  rclcpp::shutdown();
  return 0;
}
