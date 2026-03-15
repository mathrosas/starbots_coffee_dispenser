#include "object_manipulation/motion_executor.hpp"
#include "object_manipulation/motion_policy.hpp"

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <behaviortree_cpp/bt_factory.h>
#include <behaviortree_cpp/loggers/groot2_publisher.h>
#include <custom_msgs/action/deliver_coffee.hpp>
#include <custom_msgs/msg/detected_objects.hpp>
#include <moveit_msgs/msg/display_robot_state.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <std_msgs/msg/string.hpp>

#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// Program constants.
static const rclcpp::Logger LOGGER = rclcpp::get_logger("move_group_node");
static const std::string PLANNING_GROUP_ROBOT = "ur_manipulator";
static const std::string PLANNING_GROUP_GRIPPER = "gripper";
static const std::string REF_FRAME = "base_link";
static const std::string CUPHOLDER_TOPIC = "/cup_holder_detected";
static const std::string DELIVER_COFFEE_ACTION = "deliver_coffee";
static const std::string BT_STATUS_TOPIC = "/bt_node_status";
static const std::string DEFAULT_BT_XML_REL_PATH =
    "/bt_config/deliver_coffee_tree.xml";

static const std::string DEFAULT_OMPL_PIPELINE = "ompl";
static const std::string DEFAULT_PILZ_PIPELINE =
    "pilz_industrial_motion_planner";
static const std::string DEFAULT_PILZ_LIN = "LIN";
static const std::string DEFAULT_CONSTRAINED_OMPL_PLANNER =
    "KPIECEkConfigDefault";

// Fixed-cup defaults.
static constexpr double FIXED_CUP_X = 0.299;
static constexpr double FIXED_CUP_Y = 0.331;
static constexpr double FIXED_CUP_Z = 0.035;

// Motion defaults (parameterized at runtime).
static constexpr double DEFAULT_PREGRASP_Z_OFFSET = 0.20;
static constexpr double DEFAULT_APPROACH_Z_DELTA = 0.01;
static constexpr double DEFAULT_PICK_RETREAT_MULTIPLIER = 10.0;
static constexpr double DEFAULT_PLACE_RETRY_Z_STEP = 0.002;
static constexpr double DEFAULT_PRE_PLACE_EXTRA_Z = 0.066;
static constexpr double DEFAULT_INSERT_RETRY_LIFT_FACTOR = 0.5;
static constexpr int DEFAULT_PRE_PLACE_INTERNAL_ATTEMPTS = 3;
static constexpr int DEFAULT_INSERT_INTERNAL_ATTEMPTS = 8;
static constexpr double DEFAULT_INSERT_RETRY_Z_BUMP = 0.006;
static constexpr int DEFAULT_SETTLE_DELAY_MS = 2000;
static constexpr double DEFAULT_POST_PLACE_RETREAT_MULTIPLIER = 3.0;
static constexpr double DEFAULT_RECOVERY_SMALL_RETREAT_Z = 0.01;

// Watch guard defaults.
static constexpr bool DEFAULT_ENABLE_WATCH_GUARDS = true;
static constexpr double DEFAULT_JOINT_STATE_TIMEOUT_SEC = 2.0;
static constexpr double DEFAULT_START_STATE_MAX_JOINT_DELTA = 0.01;
static constexpr double DEFAULT_CUPHOLDER_MAX_AGE_SEC = 15.0;
static constexpr int DEFAULT_MIN_TRAJ_POINTS_OMPL = 2;
static constexpr int DEFAULT_MIN_TRAJ_POINTS_PILZ = 2;

// Constraint defaults.
static constexpr double DEFAULT_CONSTRAINT_X_TOL = 0.1;
static constexpr double DEFAULT_CONSTRAINT_Y_TOL = 0.1;
static constexpr double DEFAULT_CONSTRAINT_Z_TOL = M_PI;
static constexpr double DEFAULT_CONSTRAINT_WEIGHT = 1.0;

class PickAndPlacePerception {
public:
  using DetectedObject = custom_msgs::msg::DetectedObjects;
  using DeliverCoffee = custom_msgs::action::DeliverCoffee;
  using GoalHandleDeliverCoffee =
      rclcpp_action::ServerGoalHandle<DeliverCoffee>;

  PickAndPlacePerception(rclcpp::Node::SharedPtr base_node)
      : base_node_(std::move(base_node)) {
    RCLCPP_INFO(LOGGER, "Initializing Class: Pick And Place Perception...");

    rclcpp::NodeOptions node_options;
    node_options.automatically_declare_parameters_from_overrides(true);

    // Initialize move_group node.
    move_group_node_ =
        rclcpp::Node::make_shared("move_group_node", node_options);
    ensure_sim_time_true(move_group_node_);
    if (this->base_node_) {
      ensure_sim_time_true(this->base_node_);
    }

    declare_base_parameters();
    declare_motion_parameters();

    const auto ch = move_group_node_->get_parameter("ch").as_string();
    target_holder_id_ = parse_ch_to_id(ch);

    executor_.add_node(move_group_node_);
    executor_thread_ = std::thread([this]() { this->executor_.spin(); });

    // Initialize MoveGroup interfaces.
    move_group_robot_ =
        std::make_shared<moveit::planning_interface::MoveGroupInterface>(
            move_group_node_, PLANNING_GROUP_ROBOT);
    move_group_gripper_ =
        std::make_shared<moveit::planning_interface::MoveGroupInterface>(
            move_group_node_, PLANNING_GROUP_GRIPPER);

    move_group_robot_->setPoseReferenceFrame(REF_FRAME);
    move_group_robot_->setGoalPositionTolerance(0.0005);
    move_group_robot_->setGoalOrientationTolerance(0.05);

    move_group_gripper_->setGoalTolerance(0.0001);
    move_group_gripper_->setMaxVelocityScalingFactor(0.1);
    move_group_gripper_->setMaxAccelerationScalingFactor(0.05);

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

    if (!joint_model_group_robot_) {
      throw std::runtime_error("Joint model group 'ur_manipulator' not found "
                               "in current robot state.");
    }
    if (!joint_model_group_gripper_) {
      throw std::runtime_error(
          "Joint model group 'gripper' not found in current robot state.");
    }

    current_state_robot_->copyJointGroupPositions(joint_model_group_robot_,
                                                  joint_group_positions_robot_);
    current_state_gripper_->copyJointGroupPositions(
        joint_model_group_gripper_, joint_group_positions_gripper_);

    move_group_robot_->setStartStateToCurrentState();
    move_group_gripper_->setStartStateToCurrentState();

    motion_executor_ = std::make_unique<object_manipulation::MotionExecutor>(
        move_group_robot_, move_group_gripper_, joint_model_group_robot_,
        LOGGER);

    auto policies = object_manipulation::build_default_motion_policies(
        ompl_pipeline_, pilz_pipeline_, pilz_lin_planner_,
        constrained_ompl_planner_);
    for (auto &entry : policies) {
      if (entry.second.pipeline_id == pilz_pipeline_) {
        entry.second.min_trajectory_points =
            static_cast<std::size_t>(min_trajectory_points_pilz_);
      } else {
        entry.second.min_trajectory_points =
            static_cast<std::size_t>(min_trajectory_points_ompl_);
      }
    }

    motion_executor_->set_motion_policies(std::move(policies));
    motion_executor_->set_watch_guards_enabled(enable_watch_guards_);
    motion_executor_->set_joint_state_timeout_sec(joint_state_timeout_sec_);
    motion_executor_->set_start_state_max_joint_delta(
        start_state_max_joint_delta_);
    motion_executor_->set_settle_delay_ms(settle_delay_ms_);

    cupholder_sub_ = move_group_node_->create_subscription<DetectedObject>(
        CUPHOLDER_TOPIC, 10,
        std::bind(&PickAndPlacePerception::cupholder_callback, this,
                  std::placeholders::_1));

    action_server_ = rclcpp_action::create_server<DeliverCoffee>(
        this->base_node_, DELIVER_COFFEE_ACTION,
        std::bind(&PickAndPlacePerception::handle_goal, this,
                  std::placeholders::_1, std::placeholders::_2),
        std::bind(&PickAndPlacePerception::handle_cancel, this,
                  std::placeholders::_1),
        std::bind(&PickAndPlacePerception::handle_accepted, this,
                  std::placeholders::_1));

    setup_behavior_tree();

    goal_worker_thread_ = std::thread([this]() { goal_worker_loop(); });

    // Print out basic system information.
    RCLCPP_INFO(LOGGER, "Planning Frame: %s",
                move_group_robot_->getPlanningFrame().c_str());
    RCLCPP_INFO(LOGGER, "End Effector Link: %s",
                move_group_robot_->getEndEffectorLink().c_str());
    RCLCPP_INFO(LOGGER, "Target cupholder param ch='%s' -> id=%d", ch.c_str(),
                target_holder_id_);

    for (const auto &name : move_group_robot_->getJointModelGroupNames()) {
      RCLCPP_INFO(LOGGER, "Available group: %s", name.c_str());
    }

    RCLCPP_INFO(
        LOGGER,
        "Class Initialized: Pick And Place Perception (Action Server: /%s)",
        DELIVER_COFFEE_ACTION.c_str());
  }

  ~PickAndPlacePerception() {
    {
      std::lock_guard<std::mutex> lock(goal_queue_mutex_);
      goal_worker_stop_ = true;
    }
    goal_queue_cv_.notify_all();
    if (goal_worker_thread_.joinable()) {
      goal_worker_thread_.join();
    }

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
    bt_place_failed_ = false;
    bt_rotated_to_place_ = false;
    last_pre_place_retry_lift_ = 0.0;

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

  BT::NodeStatus tick_bt_once() { return bt_tree_.tickOnce(); }

  void halt_bt_tree() {
    if (bt_tree_ready_) {
      bt_tree_.haltTree();
    }
  }

private:
  // using shorthand for lengthy class references
  using MoveGroupInterface = moveit::planning_interface::MoveGroupInterface;
  using JointModelGroup = moveit::core::JointModelGroup;
  using RobotStatePtr = moveit::core::RobotStatePtr;

  struct MotionGeometryConfig {
    double pregrasp_z_offset{DEFAULT_PREGRASP_Z_OFFSET};
    double approach_z_delta{DEFAULT_APPROACH_Z_DELTA};
    double pick_retreat_multiplier{DEFAULT_PICK_RETREAT_MULTIPLIER};
    double place_retry_z_step{DEFAULT_PLACE_RETRY_Z_STEP};
    double pre_place_extra_z{DEFAULT_PRE_PLACE_EXTRA_Z};
    double insert_retry_lift_factor{DEFAULT_INSERT_RETRY_LIFT_FACTOR};
    int pre_place_internal_attempts{DEFAULT_PRE_PLACE_INTERNAL_ATTEMPTS};
    int insert_internal_attempts{DEFAULT_INSERT_INTERNAL_ATTEMPTS};
    double insert_retry_z_bump{DEFAULT_INSERT_RETRY_Z_BUMP};
    double post_place_retreat_multiplier{DEFAULT_POST_PLACE_RETREAT_MULTIPLIER};
    double recovery_small_retreat_z{DEFAULT_RECOVERY_SMALL_RETREAT_Z};
  };

  struct ConstraintConfig {
    double abs_x_tolerance{DEFAULT_CONSTRAINT_X_TOL};
    double abs_y_tolerance{DEFAULT_CONSTRAINT_Y_TOL};
    double abs_z_tolerance{DEFAULT_CONSTRAINT_Z_TOL};
    double weight{DEFAULT_CONSTRAINT_WEIGHT};
  };

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

  std::vector<double> joint_group_positions_gripper_;
  RobotStatePtr current_state_gripper_;

  std::unique_ptr<object_manipulation::MotionExecutor> motion_executor_;

  int target_holder_id_{1};
  std::string bt_xml_path_;
  bool bt_enable_groot_{true};
  BT::BehaviorTreeFactory bt_factory_;
  BT::Tree bt_tree_;
  std::unique_ptr<BT::Groot2Publisher> bt_groot_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr bt_status_pub_;
  bool bt_tree_ready_{false};
  std::string bt_failure_reason_;
  std::shared_ptr<GoalHandleDeliverCoffee> active_goal_handle_;
  uint32_t active_holder_id_{0};
  DetectedObject active_holder_;
  bool bt_goal_prepared_{false};
  bool bt_place_failed_{false};
  bool bt_rotated_to_place_{false};
  double last_pre_place_retry_lift_{0.0};

  MotionGeometryConfig motion_geometry_;
  ConstraintConfig constraint_cfg_;

  std::string ompl_pipeline_{DEFAULT_OMPL_PIPELINE};
  std::string pilz_pipeline_{DEFAULT_PILZ_PIPELINE};
  std::string pilz_lin_planner_{DEFAULT_PILZ_LIN};
  std::string constrained_ompl_planner_{DEFAULT_CONSTRAINED_OMPL_PLANNER};

  bool enable_watch_guards_{DEFAULT_ENABLE_WATCH_GUARDS};
  double joint_state_timeout_sec_{DEFAULT_JOINT_STATE_TIMEOUT_SEC};
  double start_state_max_joint_delta_{DEFAULT_START_STATE_MAX_JOINT_DELTA};
  double cupholder_max_age_sec_{DEFAULT_CUPHOLDER_MAX_AGE_SEC};
  int min_trajectory_points_ompl_{DEFAULT_MIN_TRAJ_POINTS_OMPL};
  int min_trajectory_points_pilz_{DEFAULT_MIN_TRAJ_POINTS_PILZ};
  int settle_delay_ms_{DEFAULT_SETTLE_DELAY_MS};

  double cup_x_{FIXED_CUP_X};
  double cup_y_{FIXED_CUP_Y};
  double cup_z_{FIXED_CUP_Z};
  double pre_x_{FIXED_CUP_X};
  double pre_y_{FIXED_CUP_Y};
  double pre_z_{FIXED_CUP_Z + DEFAULT_PREGRASP_Z_OFFSET};
  double place_x_{0.0};
  double place_y_{0.0};
  double place_z_{0.0};

  rclcpp::Subscription<DetectedObject>::SharedPtr cupholder_sub_;
  rclcpp_action::Server<DeliverCoffee>::SharedPtr action_server_;
  std::mutex cupholder_mutex_;
  std::unordered_map<uint32_t, DetectedObject> latest_cupholders_;
  std::unordered_map<uint32_t, rclcpp::Time> latest_cupholder_stamps_;

  std::mutex execution_mutex_;
  bool execution_active_{false};

  std::thread goal_worker_thread_;
  std::mutex goal_queue_mutex_;
  std::condition_variable goal_queue_cv_;
  std::deque<std::shared_ptr<GoalHandleDeliverCoffee>> goal_queue_;
  bool goal_worker_stop_{false};

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
      return ament_index_cpp::get_package_share_directory(
                 "object_manipulation") +
             DEFAULT_BT_XML_REL_PATH;
    } catch (const std::exception &) {
      return "deliver_coffee_tree.xml";
    }
  }

  void declare_base_parameters() {
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
      move_group_node_->declare_parameter<std::string>("bt_xml_path",
                                                       default_bt_xml_path());
    }
    if (!move_group_node_->has_parameter("bt_enable_groot")) {
      move_group_node_->declare_parameter<bool>("bt_enable_groot", true);
    }
  }

  void declare_motion_parameters() {
    const auto declare_if_needed = [this](const std::string &name,
                                          const auto &default_value) {
      if (!move_group_node_->has_parameter(name)) {
        move_group_node_->declare_parameter(name, default_value);
      }
    };

    // Geometry/motion tuning.
    declare_if_needed("motion.pregrasp_z_offset", DEFAULT_PREGRASP_Z_OFFSET);
    declare_if_needed("motion.approach_z_delta", DEFAULT_APPROACH_Z_DELTA);
    declare_if_needed("motion.pick_retreat_multiplier",
                      DEFAULT_PICK_RETREAT_MULTIPLIER);
    declare_if_needed("motion.place_retry_z_step", DEFAULT_PLACE_RETRY_Z_STEP);
    declare_if_needed("motion.pre_place_extra_z", DEFAULT_PRE_PLACE_EXTRA_Z);
    declare_if_needed("motion.insert_retry_lift_factor",
                      DEFAULT_INSERT_RETRY_LIFT_FACTOR);
    declare_if_needed("motion.pre_place_internal_attempts",
                      DEFAULT_PRE_PLACE_INTERNAL_ATTEMPTS);
    declare_if_needed("motion.insert_internal_attempts",
                      DEFAULT_INSERT_INTERNAL_ATTEMPTS);
    declare_if_needed("motion.insert_retry_z_bump",
                      DEFAULT_INSERT_RETRY_Z_BUMP);
    declare_if_needed("motion.settle_delay_ms", DEFAULT_SETTLE_DELAY_MS);
    declare_if_needed("motion.post_place_retreat_multiplier",
                      DEFAULT_POST_PLACE_RETREAT_MULTIPLIER);
    declare_if_needed("motion.recovery_small_retreat_z",
                      DEFAULT_RECOVERY_SMALL_RETREAT_Z);

    // Pipeline/planner policy tuning.
    declare_if_needed("motion.pipeline.ompl", DEFAULT_OMPL_PIPELINE);
    declare_if_needed("motion.pipeline.pilz", DEFAULT_PILZ_PIPELINE);
    declare_if_needed("motion.planner.pilz_lin", DEFAULT_PILZ_LIN);
    declare_if_needed("motion.planner.constrained_ompl",
                      DEFAULT_CONSTRAINED_OMPL_PLANNER);

    // Watch guard tuning.
    declare_if_needed("watch.enable", DEFAULT_ENABLE_WATCH_GUARDS);
    declare_if_needed("watch.joint_state_timeout_sec",
                      DEFAULT_JOINT_STATE_TIMEOUT_SEC);
    declare_if_needed("watch.start_state_max_joint_delta",
                      DEFAULT_START_STATE_MAX_JOINT_DELTA);
    declare_if_needed("watch.cupholder_max_age_sec",
                      DEFAULT_CUPHOLDER_MAX_AGE_SEC);
    declare_if_needed("watch.min_trajectory_points_ompl",
                      DEFAULT_MIN_TRAJ_POINTS_OMPL);
    declare_if_needed("watch.min_trajectory_points_pilz",
                      DEFAULT_MIN_TRAJ_POINTS_PILZ);

    // Constraint tuning.
    declare_if_needed("constraint.gripper_down.abs_x_tol",
                      DEFAULT_CONSTRAINT_X_TOL);
    declare_if_needed("constraint.gripper_down.abs_y_tol",
                      DEFAULT_CONSTRAINT_Y_TOL);
    declare_if_needed("constraint.gripper_down.abs_z_tol",
                      DEFAULT_CONSTRAINT_Z_TOL);
    declare_if_needed("constraint.gripper_down.weight",
                      DEFAULT_CONSTRAINT_WEIGHT);

    motion_geometry_.pregrasp_z_offset =
        move_group_node_->get_parameter("motion.pregrasp_z_offset").as_double();
    motion_geometry_.approach_z_delta =
        move_group_node_->get_parameter("motion.approach_z_delta").as_double();
    motion_geometry_.pick_retreat_multiplier =
        move_group_node_->get_parameter("motion.pick_retreat_multiplier")
            .as_double();
    motion_geometry_.place_retry_z_step =
        move_group_node_->get_parameter("motion.place_retry_z_step")
            .as_double();
    motion_geometry_.pre_place_extra_z =
        move_group_node_->get_parameter("motion.pre_place_extra_z").as_double();
    motion_geometry_.insert_retry_lift_factor =
        move_group_node_->get_parameter("motion.insert_retry_lift_factor")
            .as_double();
    motion_geometry_.pre_place_internal_attempts = std::max(
        1, static_cast<int>(
               move_group_node_
                   ->get_parameter("motion.pre_place_internal_attempts")
                   .as_int()));
    motion_geometry_.insert_internal_attempts = std::max(
        1,
        static_cast<int>(
            move_group_node_->get_parameter("motion.insert_internal_attempts")
                .as_int()));
    motion_geometry_.insert_retry_z_bump =
        move_group_node_->get_parameter("motion.insert_retry_z_bump")
            .as_double();
    settle_delay_ms_ = static_cast<int>(
        move_group_node_->get_parameter("motion.settle_delay_ms").as_int());
    motion_geometry_.post_place_retreat_multiplier =
        move_group_node_->get_parameter("motion.post_place_retreat_multiplier")
            .as_double();
    motion_geometry_.recovery_small_retreat_z =
        move_group_node_->get_parameter("motion.recovery_small_retreat_z")
            .as_double();

    // Keep geometric parameters physically meaningful.
    motion_geometry_.pregrasp_z_offset =
        std::max(0.0, motion_geometry_.pregrasp_z_offset);
    motion_geometry_.approach_z_delta =
        std::fabs(motion_geometry_.approach_z_delta);
    motion_geometry_.pick_retreat_multiplier =
        std::max(1.0, motion_geometry_.pick_retreat_multiplier);
    motion_geometry_.place_retry_z_step =
        std::max(0.0, motion_geometry_.place_retry_z_step);
    motion_geometry_.pre_place_extra_z =
        std::max(0.0, motion_geometry_.pre_place_extra_z);
    motion_geometry_.insert_retry_lift_factor =
        std::max(0.0, motion_geometry_.insert_retry_lift_factor);
    motion_geometry_.insert_retry_z_bump =
        std::max(0.0, motion_geometry_.insert_retry_z_bump);
    settle_delay_ms_ = std::max(0, settle_delay_ms_);
    motion_geometry_.post_place_retreat_multiplier =
        std::max(0.0, motion_geometry_.post_place_retreat_multiplier);
    motion_geometry_.recovery_small_retreat_z =
        std::max(0.0, motion_geometry_.recovery_small_retreat_z);

    ompl_pipeline_ =
        move_group_node_->get_parameter("motion.pipeline.ompl").as_string();
    pilz_pipeline_ =
        move_group_node_->get_parameter("motion.pipeline.pilz").as_string();
    pilz_lin_planner_ =
        move_group_node_->get_parameter("motion.planner.pilz_lin").as_string();
    constrained_ompl_planner_ =
        move_group_node_->get_parameter("motion.planner.constrained_ompl")
            .as_string();

    enable_watch_guards_ =
        move_group_node_->get_parameter("watch.enable").as_bool();
    joint_state_timeout_sec_ =
        move_group_node_->get_parameter("watch.joint_state_timeout_sec")
            .as_double();
    joint_state_timeout_sec_ = std::max(0.1, joint_state_timeout_sec_);
    start_state_max_joint_delta_ =
        move_group_node_->get_parameter("watch.start_state_max_joint_delta")
            .as_double();
    start_state_max_joint_delta_ = std::max(1e-4, start_state_max_joint_delta_);
    cupholder_max_age_sec_ =
        move_group_node_->get_parameter("watch.cupholder_max_age_sec")
            .as_double();
    cupholder_max_age_sec_ = std::max(0.1, cupholder_max_age_sec_);
    min_trajectory_points_ompl_ = std::max(
        1,
        static_cast<int>(
            move_group_node_->get_parameter("watch.min_trajectory_points_ompl")
                .as_int()));
    min_trajectory_points_pilz_ = std::max(
        1,
        static_cast<int>(
            move_group_node_->get_parameter("watch.min_trajectory_points_pilz")
                .as_int()));

    constraint_cfg_.abs_x_tolerance =
        move_group_node_->get_parameter("constraint.gripper_down.abs_x_tol")
            .as_double();
    constraint_cfg_.abs_y_tolerance =
        move_group_node_->get_parameter("constraint.gripper_down.abs_y_tol")
            .as_double();
    constraint_cfg_.abs_z_tolerance =
        move_group_node_->get_parameter("constraint.gripper_down.abs_z_tol")
            .as_double();
    constraint_cfg_.weight =
        move_group_node_->get_parameter("constraint.gripper_down.weight")
            .as_double();

    constraint_cfg_.abs_x_tolerance =
        std::max(1e-4, constraint_cfg_.abs_x_tolerance);
    constraint_cfg_.abs_y_tolerance =
        std::max(1e-4, constraint_cfg_.abs_y_tolerance);
    constraint_cfg_.abs_z_tolerance =
        std::max(1e-4, constraint_cfg_.abs_z_tolerance);
    constraint_cfg_.weight = std::max(0.0, constraint_cfg_.weight);

    pre_z_ = cup_z_ + motion_geometry_.pregrasp_z_offset;
  }

  void setup_behavior_tree() {
    bt_xml_path_ = move_group_node_->get_parameter("bt_xml_path").as_string();
    bt_enable_groot_ =
        move_group_node_->get_parameter("bt_enable_groot").as_bool();

    register_bt_nodes();
    load_behavior_tree(bt_xml_path_);

    bt_status_pub_ = base_node_->create_publisher<std_msgs::msg::String>(
        BT_STATUS_TOPIC, 10);
    RCLCPP_INFO(LOGGER, "BT status telemetry enabled on topic '%s'.",
                BT_STATUS_TOPIC.c_str());

    if (bt_enable_groot_) {
      try {
        bt_groot_publisher_ = std::make_unique<BT::Groot2Publisher>(bt_tree_);
        RCLCPP_INFO(LOGGER, "Groot2 publisher enabled for DeliverCoffee BT.");
      } catch (const std::exception &e) {
        RCLCPP_WARN(LOGGER, "Failed to enable Groot publisher: %s", e.what());
      }
    }
  }

  BT::NodeStatus publish_bt_node_status(const std::string &node_name,
                                        BT::NodeStatus status) {
    if (!bt_status_pub_) {
      return status;
    }
    std_msgs::msg::String msg;
    msg.data = "[BT] " + node_name + " Node -> " + BT::toStr(status, false);
    bt_status_pub_->publish(msg);
    return status;
  }

  BT::NodeStatus run_bt_node(const std::string &node_name,
                             const std::function<BT::NodeStatus()> &fn) {
    return publish_bt_node_status(node_name, fn());
  }

  void register_bt_nodes() {
    bt_factory_.registerSimpleCondition(
        "GoalNotCanceled", [this](BT::TreeNode & /*node*/) {
          return run_bt_node("GoalNotCanceled",
                             [this]() { return bt_goal_not_canceled(); });
        });

    // -sim style node IDs.
    bt_factory_.registerSimpleAction(
        "ValidateDetection", [this](BT::TreeNode & /*node*/) {
          return run_bt_node("ValidateDetection",
                             [this]() { return bt_validate_detection(); });
        });
    bt_factory_.registerSimpleAction(
        "PrePick", [this](BT::TreeNode & /*node*/) {
          return run_bt_node("PrePick", [this]() { return bt_pre_pick(); });
        });
    bt_factory_.registerSimpleAction("Pick", [this](BT::TreeNode & /*node*/) {
      return run_bt_node("Pick", [this]() { return bt_pick(); });
    });
    bt_factory_.registerSimpleAction(
        "PrePlace", [this](BT::TreeNode & /*node*/) {
          return run_bt_node("PrePlace", [this]() { return bt_pre_place(); });
        });
    bt_factory_.registerSimpleAction("Place", [this](BT::TreeNode & /*node*/) {
      return run_bt_node("Place", [this]() { return bt_place(); });
    });
    bt_factory_.registerSimpleAction(
        "PutBack", [this](BT::TreeNode & /*node*/) {
          return run_bt_node("PutBack", [this]() { return bt_put_back(); });
        });
    bt_factory_.registerSimpleAction("Return", [this](BT::TreeNode & /*node*/) {
      return run_bt_node("Return", [this]() { return bt_return(); });
    });
    bt_factory_.registerSimpleAction("ForceFail", [this](
                                                      BT::TreeNode & /*node*/) {
      return run_bt_node("ForceFail", [this]() { return bt_force_failure(); });
    });

    // Legacy node IDs kept for backward-compatible XMLs.
    bt_factory_.registerSimpleAction(
        "AcquireTarget", [this](BT::TreeNode & /*node*/) {
          return run_bt_node("AcquireTarget",
                             [this]() { return bt_acquire_target(); });
        });
    bt_factory_.registerSimpleAction(
        "MovePregrasp", [this](BT::TreeNode & /*node*/) {
          return run_bt_node("MovePregrasp",
                             [this]() { return bt_move_pregrasp(); });
        });
    bt_factory_.registerSimpleAction(
        "OpenGripper", [this](BT::TreeNode & /*node*/) {
          return run_bt_node("OpenGripper",
                             [this]() { return bt_open_gripper(); });
        });
    bt_factory_.registerSimpleAction(
        "ApproachCup", [this](BT::TreeNode & /*node*/) {
          return run_bt_node("ApproachCup",
                             [this]() { return bt_approach_cup(); });
        });
    bt_factory_.registerSimpleAction(
        "CloseGripper", [this](BT::TreeNode & /*node*/) {
          return run_bt_node("CloseGripper",
                             [this]() { return bt_close_gripper(); });
        });
    bt_factory_.registerSimpleAction(
        "RetreatWithCup", [this](BT::TreeNode & /*node*/) {
          return run_bt_node("RetreatWithCup",
                             [this]() { return bt_retreat_with_cup(); });
        });
    bt_factory_.registerSimpleAction(
        "RotateToPlace", [this](BT::TreeNode & /*node*/) {
          return run_bt_node("RotateToPlace",
                             [this]() { return bt_rotate_to_place(); });
        });
    bt_factory_.registerSimpleAction(
        "MovePrePlace", [this](BT::TreeNode & /*node*/) {
          return run_bt_node("MovePrePlace", [this]() {
            std::string reason;
            const auto status = bt_move_pre_place(0.0, 0, 1, &reason);
            if (status != BT::NodeStatus::SUCCESS) {
              return bt_fail(reason.empty() ? "Failed pre-place motion"
                                            : reason);
            }
            return BT::NodeStatus::SUCCESS;
          });
        });
    bt_factory_.registerSimpleAction(
        "InsertCup", [this](BT::TreeNode & /*node*/) {
          return run_bt_node("InsertCup", [this]() { return bt_insert_cup(); });
        });
    bt_factory_.registerSimpleAction(
        "ReleaseCup", [this](BT::TreeNode & /*node*/) {
          return run_bt_node("ReleaseCup",
                             [this]() { return bt_release_cup(); });
        });
    bt_factory_.registerSimpleAction(
        "PostPlaceRetreat", [this](BT::TreeNode & /*node*/) {
          return run_bt_node("PostPlaceRetreat",
                             [this]() { return bt_post_place_retreat(); });
        });
    bt_factory_.registerSimpleAction(
        "ReturnHome", [this](BT::TreeNode & /*node*/) {
          return run_bt_node("ReturnHome",
                             [this]() { return bt_return_home(); });
        });
    bt_factory_.registerSimpleAction(
        "GoSafePose", [this](BT::TreeNode & /*node*/) {
          return run_bt_node("GoSafePose",
                             [this]() { return bt_go_safe_pose(); });
        });
    bt_factory_.registerSimpleAction(
        "RetreatSmallZ", [this](BT::TreeNode & /*node*/) {
          return run_bt_node("RetreatSmallZ",
                             [this]() { return bt_retreat_small_z(); });
        });
    bt_factory_.registerSimpleAction(
        "PutCupBackFixed", [this](BT::TreeNode & /*node*/) {
          return run_bt_node("PutCupBackFixed",
                             [this]() { return bt_put_cup_back_fixed(); });
        });
    bt_factory_.registerSimpleAction(
        "FailAttempt", [this](BT::TreeNode & /*node*/) {
          return run_bt_node("FailAttempt",
                             [this]() { return bt_force_failure(); });
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

  BT::NodeStatus bt_validate_detection() {
    bt_place_failed_ = false;
    bt_rotated_to_place_ = false;
    last_pre_place_retry_lift_ = 0.0;
    return bt_acquire_target();
  }

  BT::NodeStatus bt_pre_pick() { return bt_move_pregrasp(); }

  BT::NodeStatus bt_pick() {
    if (bt_open_gripper() != BT::NodeStatus::SUCCESS) {
      return BT::NodeStatus::FAILURE;
    }
    if (bt_approach_cup() != BT::NodeStatus::SUCCESS) {
      return BT::NodeStatus::FAILURE;
    }
    if (bt_close_gripper() != BT::NodeStatus::SUCCESS) {
      return BT::NodeStatus::FAILURE;
    }
    if (bt_retreat_with_cup() != BT::NodeStatus::SUCCESS) {
      return BT::NodeStatus::FAILURE;
    }
    return BT::NodeStatus::SUCCESS;
  }

  BT::NodeStatus bt_pre_place() {
    if (!bt_rotated_to_place_) {
      if (bt_rotate_to_place() != BT::NodeStatus::SUCCESS) {
        return BT::NodeStatus::FAILURE;
      }
      bt_rotated_to_place_ = true;
    }

    const int attempts = motion_geometry_.pre_place_internal_attempts;
    std::string last_reason;
    for (int attempt = 0; attempt < attempts; ++attempt) {
      const double retry_lift =
          static_cast<double>(attempt) * motion_geometry_.place_retry_z_step;

      if (bt_move_pre_place(retry_lift, attempt, attempts, &last_reason) ==
          BT::NodeStatus::SUCCESS) {
        last_pre_place_retry_lift_ = retry_lift;
        return BT::NodeStatus::SUCCESS;
      }

      RCLCPP_WARN(
          LOGGER,
          "[pre_place] Internal attempt %d/%d failed (lift=%.1f mm): %s",
          attempt + 1, attempts, retry_lift * 1000.0,
          last_reason.empty() ? "unknown reason" : last_reason.c_str());
    }

    std::ostringstream ss;
    ss << "Failed pre-place after " << attempts << " internal attempts";
    if (!last_reason.empty()) {
      ss << ": " << last_reason;
    }
    return bt_fail(ss.str());
  }

  BT::NodeStatus bt_place() {
    if (bt_insert_cup() != BT::NodeStatus::SUCCESS) {
      return BT::NodeStatus::FAILURE;
    }
    if (bt_release_cup() != BT::NodeStatus::SUCCESS) {
      return BT::NodeStatus::FAILURE;
    }
    if (bt_post_place_retreat() != BT::NodeStatus::SUCCESS) {
      return BT::NodeStatus::FAILURE;
    }
    return BT::NodeStatus::SUCCESS;
  }

  BT::NodeStatus bt_put_back() {
    bt_place_failed_ = true;
    return bt_put_cup_back_fixed();
  }

  BT::NodeStatus bt_return() {
    const auto status = bt_return_home();
    if (status != BT::NodeStatus::SUCCESS) {
      return status;
    }
    if (bt_place_failed_) {
      return bt_fail("Place failed: cup was put back to fixed pick position.");
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
    pre_z_ = cup_z_ + motion_geometry_.pregrasp_z_offset;

    RCLCPP_INFO(LOGGER,
                "Using cupholder id=%u pose=(%.3f, %.3f, %.3f) w=%.3f h=%.3f "
                "t=%.3f",
                active_holder_.object_id, active_holder_.position.x,
                active_holder_.position.y, active_holder_.position.z,
                active_holder_.width, active_holder_.height,
                active_holder_.thickness);
    RCLCPP_INFO(LOGGER,
                "[PLACE_DIAG] Target cupholder id=%u pose=(%.4f, %.4f, %.4f)",
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

  geometry_msgs::msg::Pose make_pose(double x, double y, double z, double qx,
                                     double qy, double qz, double qw) const {
    geometry_msgs::msg::Pose pose;
    pose.position.x = x;
    pose.position.y = y;
    pose.position.z = z;
    pose.orientation.x = qx;
    pose.orientation.y = qy;
    pose.orientation.z = qz;
    pose.orientation.w = qw;
    return pose;
  }

  moveit_msgs::msg::Constraints build_gripper_down_constraints() const {
    return motion_executor_->make_gripper_down_constraints(
        move_group_robot_->getEndEffectorLink(),
        move_group_robot_->getPlanningFrame(), constraint_cfg_.abs_x_tolerance,
        constraint_cfg_.abs_y_tolerance, constraint_cfg_.abs_z_tolerance,
        constraint_cfg_.weight);
  }

  BT::NodeStatus bt_move_pregrasp() {
    publish_feedback(active_goal_handle_, "pregrasp", 0.10f, active_holder_id_);
    RCLCPP_INFO(LOGGER, "Going to Pregrasp Position (%.3f, %.3f, %.3f)...",
                pre_x_, pre_y_, pre_z_);

    const auto pose =
        make_pose(pre_x_, pre_y_, pre_z_, -1.000, +0.000, +0.000, +0.000);

    std::string reason;
    std::size_t points = 0;
    if (!motion_executor_->execute_pose_goal(
            object_manipulation::MotionStage::kPregrasp, pose, REF_FRAME,
            std::nullopt, reason, &points)) {
      return bt_fail(reason.empty() ? "Failed pregrasp motion" : reason);
    }

    publish_feedback(active_goal_handle_, "pregrasp_reached", 0.12f,
                     active_holder_id_);
    return BT::NodeStatus::SUCCESS;
  }

  BT::NodeStatus bt_open_gripper() {
    publish_feedback(active_goal_handle_, "open_gripper", 0.18f,
                     active_holder_id_);
    RCLCPP_INFO(LOGGER, "Opening Gripper...");

    std::string reason;
    if (!motion_executor_->execute_gripper_named("open", reason)) {
      return bt_fail(reason.empty() ? "Failed to open gripper" : reason);
    }

    publish_feedback(active_goal_handle_, "gripper_open", 0.20f,
                     active_holder_id_);
    return BT::NodeStatus::SUCCESS;
  }

  BT::NodeStatus bt_approach_cup() {
    publish_feedback(active_goal_handle_, "approach_cup", 0.30f,
                     active_holder_id_);
    RCLCPP_INFO(LOGGER, "Approaching object...");

    std::string reason;
    std::size_t points = 0;
    if (!motion_executor_->execute_linear_delta(
            object_manipulation::MotionStage::kApproachCup, +0.000, +0.000,
            -motion_geometry_.approach_z_delta, std::nullopt, reason,
            &points)) {
      return bt_fail(reason.empty() ? "Failed linear approach to cup" : reason);
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

    std::string reason;
    if (!motion_executor_->execute_gripper_named("close", reason)) {
      return bt_fail(reason.empty() ? "Failed to close gripper" : reason);
    }

    publish_feedback(active_goal_handle_, "gripper_closed", 0.42f,
                     active_holder_id_);
    return BT::NodeStatus::SUCCESS;
  }

  BT::NodeStatus bt_retreat_with_cup() {
    publish_feedback(active_goal_handle_, "retreat_with_cup", 0.50f,
                     active_holder_id_);
    RCLCPP_INFO(LOGGER, "Retreating...");

    std::string reason;
    std::size_t points = 0;
    const double z_delta = motion_geometry_.pick_retreat_multiplier *
                           motion_geometry_.approach_z_delta;
    if (!motion_executor_->execute_linear_delta(
            object_manipulation::MotionStage::kRetreatWithCup, +0.000, +0.000,
            z_delta, std::nullopt, reason, &points)) {
      return bt_fail(reason.empty() ? "Failed linear retreat" : reason);
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
    if (!current_state_robot_) {
      return bt_fail("Failed reading current robot state before rotate");
    }

    current_state_robot_->copyJointGroupPositions(joint_model_group_robot_,
                                                  joint_group_positions_robot_);
    joint_group_positions_robot_.resize(6);
    joint_group_positions_robot_[0] += 2.0 * M_PI / 3.0;

    std::string reason;
    std::size_t points = 0;
    if (!motion_executor_->execute_joint_goal(
            object_manipulation::MotionStage::kRotateToPlace,
            joint_group_positions_robot_, reason, &points)) {
      return bt_fail(reason.empty() ? "Failed shoulder rotation" : reason);
    }

    publish_feedback(active_goal_handle_, "rotated_to_place_side", 0.62f,
                     active_holder_id_);
    return BT::NodeStatus::SUCCESS;
  }

  bool ensure_cupholder_fresh(uint32_t holder_id, const std::string &stage) {
    std::lock_guard<std::mutex> lock(cupholder_mutex_);
    const auto it = latest_cupholder_stamps_.find(holder_id);
    if (it == latest_cupholder_stamps_.end()) {
      RCLCPP_ERROR(LOGGER,
                   "WatchGuard failed in %s: no timestamp for cupholder id=%u",
                   stage.c_str(), holder_id);
      return false;
    }

    const rclcpp::Duration age = move_group_node_->now() - it->second;
    if (age.seconds() > cupholder_max_age_sec_) {
      RCLCPP_ERROR(
          LOGGER,
          "WatchGuard failed in %s: cupholder id=%u is stale (age=%.3fs,"
          " max=%.3fs)",
          stage.c_str(), holder_id, age.seconds(), cupholder_max_age_sec_);
      return false;
    }

    return true;
  }

  BT::NodeStatus bt_move_pre_place(double retry_lift, int attempt_index = 0,
                                   int total_attempts = 1,
                                   std::string *attempt_reason = nullptr) {
    publish_feedback(active_goal_handle_, "pre_place", 0.72f,
                     active_holder_id_);

    if (!ensure_cupholder_fresh(active_holder_id_, "pre_place")) {
      const std::string reason = "Cupholder target is stale before pre-place";
      if (attempt_reason) {
        *attempt_reason = reason;
      }
      return BT::NodeStatus::FAILURE;
    }

    const double pre_place_z = place_z_ + motion_geometry_.pregrasp_z_offset +
                               motion_geometry_.pre_place_extra_z + retry_lift;

    RCLCPP_INFO(LOGGER, "Going to Pre-place Position (%.3f, %.3f, %.3f)...",
                place_x_, place_y_, pre_place_z);
    RCLCPP_INFO(LOGGER,
                "Pre-place internal attempt %d/%d, lift=%.1f mm (step=%.1f mm)",
                attempt_index + 1, total_attempts, retry_lift * 1000.0,
                motion_geometry_.place_retry_z_step * 1000.0);

    const auto pose = make_pose(place_x_, place_y_, pre_place_z, -1.000, +0.000,
                                +0.000, +0.000);

    std::string reason;
    std::size_t points = 0;
    if (!motion_executor_->execute_pose_goal(
            object_manipulation::MotionStage::kPrePlace, pose, REF_FRAME,
            build_gripper_down_constraints(), reason, &points)) {
      if (attempt_reason) {
        *attempt_reason =
            reason.empty() ? "Failed pre-place motion" : std::move(reason);
      }
      return BT::NodeStatus::FAILURE;
    }

    RCLCPP_INFO(LOGGER,
                "[pre_place] Trajectory has %zu points (attempt %d, z=%.3f)",
                points, attempt_index + 1, pre_place_z);

    publish_feedback(active_goal_handle_, "pre_place_reached", 0.74f,
                     active_holder_id_);
    if (attempt_reason) {
      attempt_reason->clear();
    }
    return BT::NodeStatus::SUCCESS;
  }

  BT::NodeStatus bt_insert_cup() {
    publish_feedback(active_goal_handle_, "insert_cup", 0.82f,
                     active_holder_id_);

    if (!ensure_cupholder_fresh(active_holder_id_, "insert")) {
      return bt_fail("Cupholder target is stale before insert");
    }

    const double base_insert_delta =
        motion_geometry_.approach_z_delta +
        motion_geometry_.insert_retry_lift_factor * last_pre_place_retry_lift_;
    const int attempts = motion_geometry_.insert_internal_attempts;

    std::string last_reason;
    for (int attempt = 0; attempt < attempts; ++attempt) {
      const double retry_bump =
          static_cast<double>(attempt) * motion_geometry_.insert_retry_z_bump;
      const double insert_delta =
          std::max(0.001, base_insert_delta - retry_bump);

      RCLCPP_INFO(LOGGER,
                  "Approaching down to Place Position (%.3f, %.3f, %.3f)...",
                  place_x_, place_y_,
                  place_z_ + motion_geometry_.pregrasp_z_offset +
                      motion_geometry_.pre_place_extra_z - insert_delta);
      RCLCPP_INFO(LOGGER,
                  "Insert internal attempt %d/%d, descend=%.1f mm "
                  "(base=%.1f mm, pre-place-lift=%.1f mm, bump=%.1f mm)",
                  attempt + 1, attempts, insert_delta * 1000.0,
                  base_insert_delta * 1000.0,
                  last_pre_place_retry_lift_ * 1000.0, retry_bump * 1000.0);

      std::string reason;
      std::size_t points = 0;
      if (motion_executor_->execute_linear_delta(
              object_manipulation::MotionStage::kInsertCup, +0.000, +0.000,
              -insert_delta, std::nullopt, reason, &points)) {
        log_xy_error_to_holder("after_insert", place_x_, place_y_, place_z_);
        publish_feedback(active_goal_handle_, "cup_inserted", 0.84f,
                         active_holder_id_);
        return BT::NodeStatus::SUCCESS;
      }

      last_reason = reason.empty() ? "Failed insert linear move" : reason;
      RCLCPP_WARN(LOGGER, "[insert_cup] Internal attempt %d/%d failed: %s",
                  attempt + 1, attempts, last_reason.c_str());
    }

    std::ostringstream ss;
    ss << "Failed insert linear move after " << attempts
       << " internal attempts";
    if (!last_reason.empty()) {
      ss << ": " << last_reason;
    }
    return bt_fail(ss.str());
  }

  BT::NodeStatus bt_release_cup() {
    publish_feedback(active_goal_handle_, "release_cup", 0.88f,
                     active_holder_id_);
    RCLCPP_INFO(LOGGER, "Opening Gripper...");

    std::string reason;
    if (!motion_executor_->execute_gripper_named("open", reason)) {
      return bt_fail(reason.empty() ? "Failed to release cup" : reason);
    }

    publish_feedback(active_goal_handle_, "cup_released", 0.90f,
                     active_holder_id_);
    return BT::NodeStatus::SUCCESS;
  }

  BT::NodeStatus bt_post_place_retreat() {
    publish_feedback(active_goal_handle_, "post_place_retreat", 0.94f,
                     active_holder_id_);

    const double z_delta = motion_geometry_.post_place_retreat_multiplier *
                           motion_geometry_.pregrasp_z_offset;
    RCLCPP_INFO(LOGGER, "Retreating after release by %.3f m...", z_delta);

    std::string reason;
    std::size_t points = 0;
    if (!motion_executor_->execute_linear_delta(
            object_manipulation::MotionStage::kPostPlaceRetreat, +0.000, +0.000,
            z_delta, std::nullopt, reason, &points)) {
      RCLCPP_WARN(LOGGER,
                  "Post-place retreat failed after release (best-effort): %s",
                  reason.empty() ? "unknown reason" : reason.c_str());
      publish_feedback(active_goal_handle_, "post_place_retreat_skipped", 0.95f,
                       active_holder_id_);
      return BT::NodeStatus::SUCCESS;
    }

    publish_feedback(active_goal_handle_, "post_place_retreat_done", 0.95f,
                     active_holder_id_);
    return BT::NodeStatus::SUCCESS;
  }

  BT::NodeStatus bt_return_home() {
    publish_feedback(active_goal_handle_, "return_home", 0.98f,
                     active_holder_id_);
    RCLCPP_INFO(LOGGER, "Going to Initial Position...");

    std::vector<double> home{+0.0000, -1.5708, +0.0000,
                             -1.5708, +0.0000, +0.0000};

    std::string reason;
    std::size_t points = 0;
    if (!motion_executor_->execute_joint_goal(
            object_manipulation::MotionStage::kReturnHome, home, reason,
            &points)) {
      return bt_fail(reason.empty() ? "Failed return to initial pose" : reason);
    }

    publish_feedback(active_goal_handle_, "returned_home", 1.0f,
                     active_holder_id_);
    return BT::NodeStatus::SUCCESS;
  }

  BT::NodeStatus bt_go_safe_pose() {
    RCLCPP_WARN(LOGGER, "BT recovery: going to safe pose.");

    std::vector<double> safe_pose{+0.0000, -1.5708, +0.0000,
                                  -1.5708, +0.0000, +0.0000};

    std::string reason;
    std::size_t points = 0;
    if (!motion_executor_->execute_joint_goal(
            object_manipulation::MotionStage::kRecoverySafePose, safe_pose,
            reason, &points)) {
      return bt_fail(reason.empty() ? "BT recovery failed: safe pose" : reason);
    }

    return BT::NodeStatus::SUCCESS;
  }

  BT::NodeStatus bt_retreat_small_z() {
    RCLCPP_WARN(LOGGER, "BT recovery: small +Z retreat.");

    std::string reason;
    std::size_t points = 0;
    if (!motion_executor_->execute_linear_delta(
            object_manipulation::MotionStage::kRecoveryRetreatSmallZ, +0.000,
            +0.000, motion_geometry_.recovery_small_retreat_z, std::nullopt,
            reason, &points)) {
      return bt_fail(reason.empty() ? "BT recovery failed: small Z retreat"
                                    : reason);
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
    std::string reason;
    std::size_t points = 0;

    const auto above_pose =
        make_pose(pre_x_, pre_y_, pre_z_, -1.000, +0.000, +0.000, +0.000);
    if (!motion_executor_->execute_pose_goal(
            object_manipulation::MotionStage::kPutBackAboveCup, above_pose,
            REF_FRAME, std::nullopt, reason, &points)) {
      RCLCPP_WARN(LOGGER,
                  "BT rotate recovery: failed moving above fixed cup pose: %s",
                  reason.c_str());
    }

    if (!motion_executor_->execute_linear_delta(
            object_manipulation::MotionStage::kPutBackDescend, +0.000, +0.000,
            -motion_geometry_.approach_z_delta, std::nullopt, reason,
            &points)) {
      RCLCPP_WARN(LOGGER,
                  "BT rotate recovery: failed descending to put-back: %s",
                  reason.c_str());
    }

    if (!motion_executor_->execute_gripper_named("open", reason)) {
      RCLCPP_WARN(LOGGER,
                  "BT rotate recovery: failed opening gripper at put-back: %s",
                  reason.c_str());
    }

    if (!motion_executor_->execute_linear_delta(
            object_manipulation::MotionStage::kPutBackRetreat, +0.000, +0.000,
            +motion_geometry_.approach_z_delta, std::nullopt, reason,
            &points)) {
      RCLCPP_WARN(LOGGER,
                  "BT rotate recovery: failed retreating after put-back: %s",
                  reason.c_str());
    }

    return BT::NodeStatus::SUCCESS;
  }

  BT::NodeStatus bt_force_failure() {
    bt_goal_prepared_ = false; // Reacquire target on retry attempt.
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
    {
      std::lock_guard<std::mutex> lock(goal_queue_mutex_);
      goal_queue_.push_back(goal_handle);
    }
    goal_queue_cv_.notify_one();
  }

  void goal_worker_loop() {
    while (rclcpp::ok()) {
      std::shared_ptr<GoalHandleDeliverCoffee> goal_handle;
      {
        std::unique_lock<std::mutex> lock(goal_queue_mutex_);
        goal_queue_cv_.wait(lock, [this]() {
          return goal_worker_stop_ || !goal_queue_.empty();
        });

        if (goal_worker_stop_ && goal_queue_.empty()) {
          return;
        }

        goal_handle = goal_queue_.front();
        goal_queue_.pop_front();
      }

      if (goal_handle) {
        execute_goal(goal_handle);
      }
    }
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
      latest_cupholder_stamps_[msg->object_id] = move_group_node_->now();
    }

    RCLCPP_INFO_THROTTLE(LOGGER, *move_group_node_->get_clock(), 3000,
                         "Updated cupholder id=%u at (%.3f, %.3f, %.3f)",
                         msg->object_id, msg->position.x, msg->position.y,
                         msg->position.z);
  }

  void log_xy_error_to_holder(const std::string &tag, double holder_x,
                              double holder_y, double holder_z) {
    const auto ee = motion_executor_->current_pose();
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
    const auto ee = motion_executor_->current_pose();
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
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  try {
    std::shared_ptr<rclcpp::Node> base_node =
        std::make_shared<rclcpp::Node>("pick_and_place_perception");
    PickAndPlacePerception pick_and_place(base_node);
    rclcpp::spin(base_node);
    rclcpp::shutdown();
    return 0;
  } catch (const std::exception &e) {
    RCLCPP_FATAL(rclcpp::get_logger("object_manipulation_main"),
                 "Fatal startup exception: %s", e.what());
  } catch (...) {
    RCLCPP_FATAL(rclcpp::get_logger("object_manipulation_main"),
                 "Fatal startup exception: unknown error");
  }

  rclcpp::shutdown();
  return 1;
}
