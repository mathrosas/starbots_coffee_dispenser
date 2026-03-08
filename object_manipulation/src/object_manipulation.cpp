#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <behaviortree_cpp_v3/bt_factory.h>
#include <behaviortree_cpp_v3/loggers/bt_zmq_publisher.h>
#include <custom_msgs/action/deliver_coffee.hpp>
#include <custom_msgs/msg/detected_objects.hpp>
#include <moveit_msgs/msg/display_robot_state.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>

#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <chrono>
#include <cmath>
#include <functional>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// program variables
static const rclcpp::Logger LOGGER = rclcpp::get_logger("move_group_node");
static const std::string PLANNING_GROUP_ROBOT = "ur_manipulator";
static const std::string PLANNING_GROUP_GRIPPER = "gripper";
static const std::string REF_FRAME = "base_link";
static const std::string CUPHOLDER_TOPIC = "/cup_holder_detected";
static const std::string DELIVER_COFFEE_ACTION = "deliver_coffee";
static const std::string OMPL_PIPELINE = "ompl";
static const std::string PILZ_PIPELINE = "pilz_industrial_motion_planner";
static const std::string PILZ_LIN = "LIN";
static const std::string DEFAULT_BT_XML_REL_PATH =
    "/bt_config/deliver_coffee_tree.xml";

// offsets / “magic numbers”:
static constexpr double PREGRASP_Z_OFFSET = 0.20; // 20 cm above detected object
static constexpr double APPROACH_Z_DELTA = 0.01;  // straight down

// project defaults for fixed-cup mode [0.260, 0.370, -0.007]
static constexpr double FIXED_CUP_X = 0.299; // 0.300
static constexpr double FIXED_CUP_Y = 0.331; // 0.330
static constexpr double FIXED_CUP_Z = 0.035;

class PickAndPlacePerception {
public:
  using DetectedObject = custom_msgs::msg::DetectedObjects;
  using DeliverCoffee = custom_msgs::action::DeliverCoffee;
  using GoalHandleDeliverCoffee =
      rclcpp_action::ServerGoalHandle<DeliverCoffee>;

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
    if (!move_group_node_->has_parameter("bt_xml_path")) {
      move_group_node_->declare_parameter<std::string>(
          "bt_xml_path", default_bt_xml_path());
    }
    if (!move_group_node_->has_parameter("bt_enable_groot")) {
      move_group_node_->declare_parameter<bool>("bt_enable_groot", true);
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
    move_group_robot_->setPlanningTime(10.0);
    move_group_robot_->setNumPlanningAttempts(20);
    move_group_robot_->setGoalPositionTolerance(0.0005);
    move_group_robot_->setGoalOrientationTolerance(0.05);
    move_group_robot_->setMaxVelocityScalingFactor(0.1);
    move_group_robot_->setMaxAccelerationScalingFactor(0.05);
    // move_group_robot_->setMaxVelocityScalingFactor(0.08);
    // move_group_robot_->setMaxAccelerationScalingFactor(0.03);

    move_group_gripper_->setGoalTolerance(0.0001);
    move_group_gripper_->setMaxVelocityScalingFactor(
        0.1); // Slow for precise/less jittery close
    move_group_gripper_->setMaxAccelerationScalingFactor(0.05);
    // move_group_gripper_->setMaxVelocityScalingFactor(
    //     0.01); // Slow for precise/less jittery close
    // move_group_gripper_->setMaxAccelerationScalingFactor(0.01);

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

    cupholder_sub_ = move_group_node_->create_subscription<DetectedObject>(
        CUPHOLDER_TOPIC, 10,
        std::bind(&PickAndPlacePerception::cupholder_callback, this,
                  std::placeholders::_1));

    action_server_ = rclcpp_action::create_server<DeliverCoffee>(
        base_node_, DELIVER_COFFEE_ACTION,
        std::bind(&PickAndPlacePerception::handle_goal, this,
                  std::placeholders::_1, std::placeholders::_2),
        std::bind(&PickAndPlacePerception::handle_cancel, this,
                  std::placeholders::_1),
        std::bind(&PickAndPlacePerception::handle_accepted, this,
                  std::placeholders::_1));
    setup_behavior_tree();

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
    RCLCPP_INFO(
        LOGGER,
        "Class Initialized: Pick And Place Perception (Action Server: /%s)",
        DELIVER_COFFEE_ACTION.c_str());
  }

  ~PickAndPlacePerception() {
    executor_.cancel();
    if (executor_thread_.joinable()) {
      executor_thread_.join();
    }
    RCLCPP_INFO(LOGGER, "Class Terminated: Pick And Place Perception");
  }

  bool execute_trajectory_plan(
      uint32_t holder_id,
      const std::shared_ptr<GoalHandleDeliverCoffee> &goal_handle,
      std::string &failure_reason) {
    if (!bt_tree_ready_) {
      failure_reason = "Behavior tree not initialized";
      return false;
    }

    active_goal_handle_ = goal_handle;
    active_holder_id_ = holder_id;
    bt_failure_reason_.clear();
    bt_goal_prepared_ = false;

    halt_bt_tree();
    BT::NodeStatus status = BT::NodeStatus::RUNNING;
    while (rclcpp::ok() && status == BT::NodeStatus::RUNNING) {
      status = tick_bt_once();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    halt_bt_tree();

    if (status == BT::NodeStatus::SUCCESS) {
      active_goal_handle_.reset();
      RCLCPP_INFO(LOGGER, "Pick And Place Perception Execution Complete");
      return true;
    }

    if (is_cancel_requested(goal_handle) ||
        bt_failure_reason_ == "Goal canceled") {
      failure_reason = "Goal canceled";
      active_goal_handle_.reset();
      return false;
    }

    failure_reason = bt_failure_reason_.empty() ? "Behavior tree failed"
                                                : bt_failure_reason_;
    active_goal_handle_.reset();
    return false;
  }

  BT::NodeStatus tick_bt_once() {
    return bt_tree_.tickRoot();
  }

  void halt_bt_tree() {
    if (bt_tree_.rootNode()) {
      bt_tree_.rootNode()->halt();
    }
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

  Plan cartesian_lin_plan_;
  bool plan_success_cartesian_{false};

  int target_holder_id_{1};
  std::string bt_xml_path_;
  bool bt_enable_groot_{true};
  BT::BehaviorTreeFactory bt_factory_;
  BT::Tree bt_tree_;
  std::unique_ptr<BT::PublisherZMQ> bt_groot_publisher_;
  bool bt_tree_ready_{false};
  std::string bt_failure_reason_;
  std::shared_ptr<GoalHandleDeliverCoffee> active_goal_handle_;
  uint32_t active_holder_id_{0};
  DetectedObject active_holder_;
  bool bt_goal_prepared_{false};

  double cup_x_{FIXED_CUP_X};
  double cup_y_{FIXED_CUP_Y};
  double cup_z_{FIXED_CUP_Z};
  double pre_x_{FIXED_CUP_X};
  double pre_y_{FIXED_CUP_Y};
  double pre_z_{FIXED_CUP_Z + PREGRASP_Z_OFFSET};
  double place_x_{0.0};
  double place_y_{0.0};
  double place_z_{0.0};

  rclcpp::Subscription<DetectedObject>::SharedPtr cupholder_sub_;
  rclcpp_action::Server<DeliverCoffee>::SharedPtr action_server_;
  std::mutex cupholder_mutex_;
  std::unordered_map<uint32_t, DetectedObject> latest_cupholders_;
  std::mutex execution_mutex_;
  bool execution_active_{false};

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

  static std::string default_bt_xml_path() {
    try {
      return ament_index_cpp::get_package_share_directory("object_manipulation") +
             DEFAULT_BT_XML_REL_PATH;
    } catch (const std::exception &) {
      return "deliver_coffee_tree.xml";
    }
  }

  void setup_behavior_tree() {
    bt_xml_path_ = move_group_node_->get_parameter("bt_xml_path").as_string();
    bt_enable_groot_ = move_group_node_->get_parameter("bt_enable_groot").as_bool();

    register_bt_nodes();
    load_behavior_tree(bt_xml_path_);

    if (bt_enable_groot_) {
      try {
        bt_groot_publisher_ = std::make_unique<BT::PublisherZMQ>(bt_tree_);
        RCLCPP_INFO(LOGGER,
                    "Groot (ZMQ) publisher enabled for DeliverCoffee BT.");
      } catch (const std::exception &e) {
        RCLCPP_WARN(LOGGER, "Failed to enable Groot publisher: %s", e.what());
      }
    }
  }

  void register_bt_nodes() {
    bt_factory_.registerSimpleCondition(
        "GoalNotCanceled", [this](BT::TreeNode & /*node*/) {
          return bt_goal_not_canceled();
        });

    bt_factory_.registerSimpleAction("AcquireTarget",
                                     [this](BT::TreeNode & /*node*/) {
                                       return bt_acquire_target();
                                     });
    bt_factory_.registerSimpleAction("MovePregrasp",
                                     [this](BT::TreeNode & /*node*/) {
                                       return bt_move_pregrasp();
                                     });
    bt_factory_.registerSimpleAction("OpenGripper",
                                     [this](BT::TreeNode & /*node*/) {
                                       return bt_open_gripper();
                                     });
    bt_factory_.registerSimpleAction("ApproachCup",
                                     [this](BT::TreeNode & /*node*/) {
                                       return bt_approach_cup();
                                     });
    bt_factory_.registerSimpleAction("CloseGripper",
                                     [this](BT::TreeNode & /*node*/) {
                                       return bt_close_gripper();
                                     });
    bt_factory_.registerSimpleAction("RetreatWithCup",
                                     [this](BT::TreeNode & /*node*/) {
                                       return bt_retreat_with_cup();
                                     });
    bt_factory_.registerSimpleAction("RotateToPlace",
                                     [this](BT::TreeNode & /*node*/) {
                                       return bt_rotate_to_place();
                                     });
    bt_factory_.registerSimpleAction("MovePrePlace",
                                     [this](BT::TreeNode & /*node*/) {
                                       return bt_move_pre_place();
                                     });
    bt_factory_.registerSimpleAction("InsertCup",
                                     [this](BT::TreeNode & /*node*/) {
                                       return bt_insert_cup();
                                     });
    bt_factory_.registerSimpleAction("ReleaseCup",
                                     [this](BT::TreeNode & /*node*/) {
                                       return bt_release_cup();
                                     });
    bt_factory_.registerSimpleAction("PostPlaceRetreat",
                                     [this](BT::TreeNode & /*node*/) {
                                       return bt_post_place_retreat();
                                     });
    bt_factory_.registerSimpleAction("ReturnHome",
                                     [this](BT::TreeNode & /*node*/) {
                                       return bt_return_home();
                                     });
    bt_factory_.registerSimpleAction("GoSafePose",
                                     [this](BT::TreeNode & /*node*/) {
                                       return bt_go_safe_pose();
                                     });
    bt_factory_.registerSimpleAction("RetreatSmallZ",
                                     [this](BT::TreeNode & /*node*/) {
                                       return bt_retreat_small_z();
                                     });
    bt_factory_.registerSimpleAction("PutCupBackFixed",
                                     [this](BT::TreeNode & /*node*/) {
                                       return bt_put_cup_back_fixed();
                                     });
    bt_factory_.registerSimpleAction("FailAttempt",
                                     [this](BT::TreeNode & /*node*/) {
                                       return bt_force_failure();
                                     });
  }

  void load_behavior_tree(const std::string &xml_path) {
    std::ifstream ifs(xml_path);
    if (!ifs.good()) {
      throw std::runtime_error("Behavior tree XML not found at: " + xml_path);
    }

    bt_tree_ = bt_factory_.createTreeFromFile(xml_path);
    bt_tree_ready_ = true;
    RCLCPP_INFO(LOGGER, "Loaded DeliverCoffee BT XML: %s", xml_path.c_str());
  }

  BT::NodeStatus bt_fail(const std::string &reason) {
    bt_failure_reason_ = reason;
    RCLCPP_ERROR(LOGGER, "%s", reason.c_str());
    return BT::NodeStatus::FAILURE;
  }

  BT::NodeStatus bt_goal_not_canceled() {
    if (is_cancel_requested(active_goal_handle_)) {
      return bt_fail("Goal canceled");
    }
    return BT::NodeStatus::SUCCESS;
  }

  BT::NodeStatus bt_acquire_target() {
    if (bt_goal_prepared_) {
      return BT::NodeStatus::SUCCESS;
    }

    publish_feedback(active_goal_handle_, "waiting_for_target_cupholder", 0.02f,
                     active_holder_id_);
    if (!wait_for_cupholder(active_holder_id_, active_holder_,
                            std::chrono::seconds(30), active_goal_handle_)) {
      if (is_cancel_requested(active_goal_handle_)) {
        return bt_fail("Goal canceled");
      }
      return bt_fail("Timed out waiting for requested cupholder detection");
    }

    place_x_ = active_holder_.position.x;
    place_y_ = active_holder_.position.y;
    place_z_ = active_holder_.position.z;

    cup_x_ = move_group_node_->get_parameter("cup_x").as_double();
    cup_y_ = move_group_node_->get_parameter("cup_y").as_double();
    cup_z_ = move_group_node_->get_parameter("cup_z").as_double();
    pre_x_ = cup_x_;
    pre_y_ = cup_y_;
    pre_z_ = cup_z_ + PREGRASP_Z_OFFSET;

    RCLCPP_INFO(LOGGER,
                "Using cupholder id=%u pose=(%.3f, %.3f, %.3f) w=%.3f h=%.3f "
                "t=%.3f",
                active_holder_.object_id, active_holder_.position.x,
                active_holder_.position.y, active_holder_.position.z,
                active_holder_.width, active_holder_.height,
                active_holder_.thickness);
    RCLCPP_INFO(
        LOGGER, "[PLACE_DIAG] Target cupholder id=%u pose=(%.4f, %.4f, %.4f)",
        active_holder_.object_id, active_holder_.position.x,
        active_holder_.position.y, active_holder_.position.z);
    RCLCPP_INFO(LOGGER, "Using fixed cup pose at (%.3f, %.3f, %.3f)", cup_x_,
                cup_y_, cup_z_);
    RCLCPP_INFO(LOGGER, "Planning and Executing Pick And Place Perception...");
    publish_feedback(active_goal_handle_, "start_delivery", 0.05f,
                     active_holder_id_);

    bt_goal_prepared_ = true;
    return BT::NodeStatus::SUCCESS;
  }

  BT::NodeStatus bt_move_pregrasp() {
    publish_feedback(active_goal_handle_, "pregrasp", 0.10f, active_holder_id_);
    RCLCPP_INFO(LOGGER, "Going to Pregrasp Position (%.3f, %.3f, %.3f)...",
                pre_x_, pre_y_, pre_z_);
    setup_goal_pose_target(pre_x_, pre_y_, pre_z_, -1.000, +0.000, +0.000,
                           +0.000);
    plan_trajectory_kinematics();
    if (!execute_trajectory_kinematics()) {
      return bt_fail("Failed pregrasp kinematics");
    }
    publish_feedback(active_goal_handle_, "pregrasp_reached", 0.12f,
                     active_holder_id_);
    return BT::NodeStatus::SUCCESS;
  }

  BT::NodeStatus bt_open_gripper() {
    publish_feedback(active_goal_handle_, "open_gripper", 0.18f,
                     active_holder_id_);
    RCLCPP_INFO(LOGGER, "Opening Gripper...");
    setup_named_pose_gripper("open");
    plan_trajectory_gripper();
    if (!execute_trajectory_gripper()) {
      return bt_fail("Failed to open gripper");
    }
    publish_feedback(active_goal_handle_, "gripper_open", 0.20f,
                     active_holder_id_);
    return BT::NodeStatus::SUCCESS;
  }

  BT::NodeStatus bt_approach_cup() {
    publish_feedback(active_goal_handle_, "approach_cup", 0.30f,
                     active_holder_id_);
    RCLCPP_INFO(LOGGER, "Approaching object...");
    setup_waypoints_target(+0.000, +0.000, -APPROACH_Z_DELTA);
    plan_trajectory_cartesian();
    if (!execute_trajectory_cartesian()) {
      return bt_fail("Failed Cartesian approach to cup");
    }
    publish_feedback(active_goal_handle_, "approached_cup", 0.32f,
                     active_holder_id_);
    return BT::NodeStatus::SUCCESS;
  }

  BT::NodeStatus bt_close_gripper() {
    publish_feedback(active_goal_handle_, "close_gripper", 0.40f,
                     active_holder_id_);
    log_xy_error_to_point("before_grasp_close", cup_x_, cup_y_, cup_z_);
    RCLCPP_INFO(LOGGER, "Closing Gripper...");
    setup_named_pose_gripper("close");
    plan_trajectory_gripper();
    if (!execute_trajectory_gripper()) {
      return bt_fail("Failed to close gripper");
    }
    publish_feedback(active_goal_handle_, "gripper_closed", 0.42f,
                     active_holder_id_);
    return BT::NodeStatus::SUCCESS;
  }

  BT::NodeStatus bt_retreat_with_cup() {
    publish_feedback(active_goal_handle_, "retreat_with_cup", 0.50f,
                     active_holder_id_);
    RCLCPP_INFO(LOGGER, "Retreating...");
    setup_waypoints_target(+0.000, +0.000, 5 * APPROACH_Z_DELTA);
    plan_trajectory_cartesian();
    if (!execute_trajectory_cartesian()) {
      return bt_fail("Failed Cartesian retreat");
    }
    publish_feedback(active_goal_handle_, "retreated_with_cup", 0.52f,
                     active_holder_id_);
    return BT::NodeStatus::SUCCESS;
  }

  BT::NodeStatus bt_rotate_to_place() {
    publish_feedback(active_goal_handle_, "rotate_to_place", 0.60f,
                     active_holder_id_);
    RCLCPP_INFO(LOGGER, "Rotate Shoulder Joint 120 degrees...");
    current_state_robot_ = move_group_robot_->getCurrentState(10);
    current_state_robot_->copyJointGroupPositions(joint_model_group_robot_,
                                                  joint_group_positions_robot_);
    setup_joint_value_target(joint_group_positions_robot_[0] + 2.0 * M_PI / 3.0,
                             joint_group_positions_robot_[1],
                             joint_group_positions_robot_[2],
                             joint_group_positions_robot_[3],
                             joint_group_positions_robot_[4],
                             joint_group_positions_robot_[5]);
    plan_trajectory_kinematics();
    if (!execute_trajectory_kinematics()) {
      return bt_fail("Failed shoulder rotation");
    }
    publish_feedback(active_goal_handle_, "rotated_to_place_side", 0.62f,
                     active_holder_id_);
    return BT::NodeStatus::SUCCESS;
  }

  BT::NodeStatus bt_move_pre_place() {
    publish_feedback(active_goal_handle_, "pre_place", 0.72f, active_holder_id_);
    RCLCPP_INFO(LOGGER, "Going to Pre-place Position (%.3f, %.3f, %.3f)...",
                place_x_, place_y_, place_z_ + PREGRASP_Z_OFFSET + 0.066);
    setup_goal_pose_target(place_x_, place_y_,
                           place_z_ + PREGRASP_Z_OFFSET + 0.066, -1.000, +0.000,
                           +0.000, +0.000);
    plan_trajectory_kinematics();
    if (!execute_trajectory_kinematics()) {
      return bt_fail("Failed pre-place kinematics");
    }
    publish_feedback(active_goal_handle_, "pre_place_reached", 0.74f,
                     active_holder_id_);
    return BT::NodeStatus::SUCCESS;
  }

  BT::NodeStatus bt_insert_cup() {
    publish_feedback(active_goal_handle_, "insert_cup", 0.82f, active_holder_id_);
    RCLCPP_INFO(LOGGER,
                "Approaching down to Place Position (%.3f, %.3f, %.3f)...",
                place_x_, place_y_,
                place_z_ + PREGRASP_Z_OFFSET + 0.066 - APPROACH_Z_DELTA);
    setup_waypoints_target(+0.000, +0.000, -APPROACH_Z_DELTA);
    plan_trajectory_cartesian();
    if (!execute_trajectory_cartesian()) {
      return bt_fail("Failed Cartesian insert to cupholder");
    }
    log_xy_error_to_holder("after_insert_cartesian", place_x_, place_y_, place_z_);
    publish_feedback(active_goal_handle_, "cup_inserted", 0.84f,
                     active_holder_id_);
    return BT::NodeStatus::SUCCESS;
  }

  BT::NodeStatus bt_release_cup() {
    publish_feedback(active_goal_handle_, "release_cup", 0.88f, active_holder_id_);
    RCLCPP_INFO(LOGGER, "Opening Gripper...");
    setup_named_pose_gripper("open");
    plan_trajectory_gripper();
    if (!execute_trajectory_gripper()) {
      return bt_fail("Failed to release cup");
    }
    publish_feedback(active_goal_handle_, "cup_released", 0.90f,
                     active_holder_id_);
    return BT::NodeStatus::SUCCESS;
  }

  BT::NodeStatus bt_post_place_retreat() {
    publish_feedback(active_goal_handle_, "post_place_retreat", 0.94f,
                     active_holder_id_);
    RCLCPP_INFO(LOGGER,
                "Going to Pre-place Position Again (%.3f, %.3f, %.3f)...",
                place_x_, place_y_, place_z_ + 3 * PREGRASP_Z_OFFSET);
    setup_goal_pose_target(place_x_, place_y_, place_z_ + 3 * PREGRASP_Z_OFFSET,
                           -1.000, +0.000, +0.000, +0.000);
    plan_trajectory_kinematics();
    if (!execute_trajectory_kinematics()) {
      return bt_fail("Failed post-place retreat");
    }
    publish_feedback(active_goal_handle_, "post_place_retreat_done", 0.95f,
                     active_holder_id_);
    return BT::NodeStatus::SUCCESS;
  }

  BT::NodeStatus bt_return_home() {
    publish_feedback(active_goal_handle_, "return_home", 0.98f, active_holder_id_);
    RCLCPP_INFO(LOGGER, "Going to Initial Position...");
    setup_joint_value_target(+0.0000, -1.5708, +0.0000, -1.5708, +0.0000,
                             +0.0000);
    plan_trajectory_kinematics();
    if (!execute_trajectory_kinematics()) {
      return bt_fail("Failed return to initial pose");
    }
    publish_feedback(active_goal_handle_, "returned_home", 1.0f,
                     active_holder_id_);
    return BT::NodeStatus::SUCCESS;
  }

  BT::NodeStatus bt_go_safe_pose() {
    RCLCPP_WARN(LOGGER, "BT recovery: going to safe pose.");
    setup_joint_value_target(+0.0000, -1.5708, +0.0000, -1.5708, +0.0000,
                             +0.0000);
    plan_trajectory_kinematics();
    if (!execute_trajectory_kinematics()) {
      return bt_fail("BT recovery failed: could not reach safe pose");
    }
    return BT::NodeStatus::SUCCESS;
  }

  BT::NodeStatus bt_retreat_small_z() {
    RCLCPP_WARN(LOGGER, "BT recovery: small +Z retreat.");
    setup_waypoints_target(+0.000, +0.000, +0.01);
    plan_trajectory_cartesian();
    if (!execute_trajectory_cartesian()) {
      return bt_fail("BT recovery failed: small Z retreat failed");
    }
    return BT::NodeStatus::SUCCESS;
  }

  BT::NodeStatus bt_put_cup_back_fixed() {
    if (is_cancel_requested(active_goal_handle_)) {
      return bt_fail("Goal canceled");
    }

    RCLCPP_WARN(LOGGER,
                "BT rotate recovery: returning cup to fixed cup position.");
    publish_feedback(active_goal_handle_, "rotate_recovery_putback", 0.66f,
                     active_holder_id_);

    // Best-effort put-back routine to avoid carrying the cup into next retry.
    setup_goal_pose_target(pre_x_, pre_y_, pre_z_, -1.000, +0.000, +0.000,
                           +0.000);
    plan_trajectory_kinematics();
    if (!execute_trajectory_kinematics()) {
      RCLCPP_WARN(LOGGER,
                  "BT rotate recovery: failed moving above fixed cup pose.");
    }

    setup_waypoints_target(+0.000, +0.000, -APPROACH_Z_DELTA);
    plan_trajectory_cartesian();
    if (!execute_trajectory_cartesian()) {
      RCLCPP_WARN(LOGGER, "BT rotate recovery: failed descending to put-back.");
    }

    setup_named_pose_gripper("open");
    plan_trajectory_gripper();
    if (!execute_trajectory_gripper()) {
      RCLCPP_WARN(LOGGER,
                  "BT rotate recovery: failed opening gripper at put-back.");
    }

    setup_waypoints_target(+0.000, +0.000, +APPROACH_Z_DELTA);
    plan_trajectory_cartesian();
    if (!execute_trajectory_cartesian()) {
      RCLCPP_WARN(LOGGER, "BT rotate recovery: failed retreating after put-back.");
    }

    return BT::NodeStatus::SUCCESS;
  }

  BT::NodeStatus bt_force_failure() {
    bt_goal_prepared_ = false; // reacquire target on retry attempt
    RCLCPP_WARN(LOGGER, "BT: forcing attempt failure to trigger full retry.");
    return BT::NodeStatus::FAILURE;
  }

  rclcpp_action::GoalResponse
  handle_goal(const rclcpp_action::GoalUUID &,
              std::shared_ptr<const DeliverCoffee::Goal> goal) {
    if (!goal || goal->cupholder_id == 0) {
      RCLCPP_WARN(LOGGER,
                  "Rejecting DeliverCoffee goal: cupholder_id must be > 0.");
      return rclcpp_action::GoalResponse::REJECT;
    }
    {
      std::lock_guard<std::mutex> lock(execution_mutex_);
      if (execution_active_) {
        RCLCPP_WARN(
            LOGGER,
            "Rejecting DeliverCoffee goal for cupholder_id=%u: server busy.",
            goal->cupholder_id);
        return rclcpp_action::GoalResponse::REJECT;
      }
      execution_active_ = true;
    }
    RCLCPP_INFO(LOGGER, "Accepted DeliverCoffee goal for cupholder_id=%u",
                goal->cupholder_id);
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse
  handle_cancel(const std::shared_ptr<GoalHandleDeliverCoffee> goal_handle) {
    (void)goal_handle;
    RCLCPP_INFO(LOGGER, "Cancel request received for DeliverCoffee");
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void
  handle_accepted(const std::shared_ptr<GoalHandleDeliverCoffee> goal_handle) {
    std::thread([this, goal_handle]() { execute_goal(goal_handle); }).detach();
  }

  void
  execute_goal(const std::shared_ptr<GoalHandleDeliverCoffee> goal_handle) {
    const auto goal = goal_handle->get_goal();
    auto result = std::make_shared<DeliverCoffee::Result>();

    const auto holder_id = goal->cupholder_id;
    std::string failure_reason;
    bool ok = false;
    try {
      ok = execute_trajectory_plan(holder_id, goal_handle, failure_reason);
    } catch (const std::exception &e) {
      failure_reason = std::string("Unhandled exception: ") + e.what();
      ok = false;
    } catch (...) {
      failure_reason = "Unhandled non-standard exception";
      ok = false;
    }

    if (goal_handle->is_canceling() || failure_reason == "Goal canceled") {
      result->success = false;
      result->message = "DeliverCoffee canceled";
      goal_handle->canceled(result);
      RCLCPP_INFO(LOGGER, "DeliverCoffee canceled for cupholder_id=%u",
                  holder_id);
      clear_execution_active();
      return;
    }

    if (ok) {
      result->success = true;
      result->message = "DeliverCoffee completed successfully";
      goal_handle->succeed(result);
      RCLCPP_INFO(LOGGER, "DeliverCoffee succeeded for cupholder_id=%u",
                  holder_id);
      clear_execution_active();
      return;
    }

    result->success = false;
    result->message =
        failure_reason.empty() ? "DeliverCoffee failed" : failure_reason;
    goal_handle->abort(result);
    RCLCPP_ERROR(LOGGER, "DeliverCoffee aborted for cupholder_id=%u: %s",
                 holder_id, result->message.c_str());
    clear_execution_active();
  }

  void clear_execution_active() {
    std::lock_guard<std::mutex> lock(execution_mutex_);
    execution_active_ = false;
  }

  bool wait_for_cupholder(
      uint32_t holder_id, DetectedObject &holder, std::chrono::seconds timeout,
      const std::shared_ptr<GoalHandleDeliverCoffee> &goal_handle = nullptr) {
    const auto start = std::chrono::steady_clock::now();
    while (rclcpp::ok()) {
      if (is_cancel_requested(goal_handle)) {
        return false;
      }
      {
        std::lock_guard<std::mutex> lock(cupholder_mutex_);
        const auto it = latest_cupholders_.find(holder_id);
        if (it != latest_cupholders_.end()) {
          holder = it->second;
          return true;
        }
      }

      if (std::chrono::steady_clock::now() - start > timeout) {
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return false;
  }

  bool is_cancel_requested(
      const std::shared_ptr<GoalHandleDeliverCoffee> &goal_handle) const {
    return goal_handle && goal_handle->is_canceling();
  }

  void
  publish_feedback(const std::shared_ptr<GoalHandleDeliverCoffee> &goal_handle,
                   const std::string &stage, float progress,
                   uint32_t holder_id) {
    if (!goal_handle) {
      return;
    }
    auto feedback = std::make_shared<DeliverCoffee::Feedback>();
    feedback->stage = stage;
    feedback->progress = progress;
    feedback->cupholder_id = holder_id;
    goal_handle->publish_feedback(feedback);
  }

  void cupholder_callback(const DetectedObject::SharedPtr msg) {
    if (!msg) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(cupholder_mutex_);
      latest_cupholders_[msg->object_id] = *msg;
    }
    RCLCPP_INFO_THROTTLE(LOGGER, *move_group_node_->get_clock(), 3000,
                         "Updated cupholder id=%u at (%.3f, %.3f, %.3f)",
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
    // OMPL goals are provided in perception/base coordinates.
    move_group_robot_->setPoseReferenceFrame(REF_FRAME);
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
    move_group_robot_->setPlanningPipelineId(OMPL_PIPELINE);
    move_group_robot_->setPlannerId("RRTConnectkConfigDefault");
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
    // Pilz LIN requires goal frame == model/planning frame (typically "world").
    move_group_robot_->setPoseReferenceFrame(
        move_group_robot_->getPlanningFrame());
    target_pose_robot_ = move_group_robot_->getCurrentPose().pose;
    target_pose_robot_.position.x += x_delta;
    target_pose_robot_.position.y += y_delta;
    target_pose_robot_.position.z += z_delta;
    move_group_robot_->setStartStateToCurrentState();
    move_group_robot_->setPoseTarget(target_pose_robot_);
  }

  void plan_trajectory_cartesian() {
    // Use Pilz LIN for deterministic straight-line TCP moves.
    move_group_robot_->setPoseReferenceFrame(
        move_group_robot_->getPlanningFrame());
    move_group_robot_->setPlanningPipelineId(PILZ_PIPELINE);
    move_group_robot_->setPlannerId(PILZ_LIN);
    plan_success_cartesian_ = (move_group_robot_->plan(cartesian_lin_plan_) ==
                               moveit::core::MoveItErrorCode::SUCCESS);
  }

  bool execute_trajectory_cartesian() {
    if (plan_success_cartesian_) {
      log_ee_and_joints("pre_execute_cartesian");
      auto code = move_group_robot_->execute(cartesian_lin_plan_);
      if (code != moveit::core::MoveItErrorCode::SUCCESS) {
        RCLCPP_ERROR(LOGGER, "Robot LIN Trajectory Execute Failed !");
        return false;
      }
      RCLCPP_INFO(LOGGER, "Robot LIN Trajectory Success !");
      log_ee_and_joints("post_execute_cartesian");
      return true;
    }
    RCLCPP_ERROR(LOGGER, "Robot LIN Trajectory Planning Failed !");
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

  void log_xy_error_to_holder(const std::string &tag, double holder_x,
                              double holder_y, double holder_z) {
    const auto ee = move_group_robot_->getCurrentPose().pose;
    const double dx = ee.position.x - holder_x;
    const double dy = ee.position.y - holder_y;
    const double dz = ee.position.z - holder_z;
    const double xy_mm = 1000.0 * std::hypot(dx, dy);

    RCLCPP_INFO(LOGGER,
                "[PLACE_DIAG:%s] holder=(%.4f, %.4f, %.4f) "
                "ee=(%.4f, %.4f, %.4f) d=(%.4f, %.4f, %.4f) |xy|=%.2f mm",
                tag.c_str(), holder_x, holder_y, holder_z, ee.position.x,
                ee.position.y, ee.position.z, dx, dy, dz, xy_mm);
  }

  void log_xy_error_to_point(const std::string &tag, double target_x,
                             double target_y, double target_z) {
    const auto ee = move_group_robot_->getCurrentPose().pose;
    const double dx = ee.position.x - target_x;
    const double dy = ee.position.y - target_y;
    const double dz = ee.position.z - target_z;
    const double xy_mm = 1000.0 * std::hypot(dx, dy);

    RCLCPP_INFO(LOGGER,
                "[PICK_DIAG:%s] target=(%.4f, %.4f, %.4f) "
                "ee=(%.4f, %.4f, %.4f) d=(%.4f, %.4f, %.4f) |xy|=%.2f mm",
                tag.c_str(), target_x, target_y, target_z, ee.position.x,
                ee.position.y, ee.position.z, dx, dy, dz, xy_mm);
  }
}; // class PickAndPlacePerception

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  std::shared_ptr<rclcpp::Node> base_node =
      std::make_shared<rclcpp::Node>("pick_and_place_perception");
  PickAndPlacePerception pick_and_place(base_node);
  rclcpp::spin(base_node);
  rclcpp::shutdown();
  return 0;
}
