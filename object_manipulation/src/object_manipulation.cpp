#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

#include <moveit_msgs/msg/robot_trajectory.hpp>

#include <custom_msgs/msg/detected_objects.hpp>

#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>

#include <chrono>
#include <cmath>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

// --------------------
// Program configuration
// --------------------
static const rclcpp::Logger LOGGER = rclcpp::get_logger("object_manipulation");

// MoveIt groups (must match your SRDF)
static const std::string PLANNING_GROUP_ROBOT = "ur_manipulator";
static const std::string PLANNING_GROUP_GRIPPER = "gripper";

// Cupholder detection (your python node publishes this)
static const std::string CUPHOLDER_TOPIC = "/cup_holder_detected";

// Frames (your python publishes everything in base_link)
static const std::string REF_FRAME = "base_link";

// -------- Fixed cup pose (first test: no cup detection) --------
static constexpr double FIXED_CUP_X = 0.318;
static constexpr double FIXED_CUP_Y = 0.350;
static constexpr double FIXED_CUP_Z = 0.035;

// -------- Fixed place pose (first test: place is also fixed) --------
// Use your known good drop location for now (adjust if needed)
static constexpr double FIXED_PLACE_X = -0.3400;
static constexpr double FIXED_PLACE_Y = -0.0045;
static constexpr double FIXED_PLACE_Z_PRE = -0.1861; // above
static constexpr double FIXED_PLACE_Z_DROP =
    -0.5861; // down (as in your script)

// -------- Motion tuning --------
static constexpr double PREGRASP_Z_OFFSET = 0.30; // 30cm above cup
static constexpr double APPROACH_Z_DELTA = -0.12; // down 12cm (pick)
static constexpr double RETREAT_Z_DELTA = +0.30;  // up 30cm (pick retreat)

// Cartesian planning
static constexpr double EEF_STEP = 0.01;
static constexpr double JUMP_THRESHOLD = 0.0;
static constexpr double CARTESIAN_MIN_FRACTION = 0.90;

// Z-down quaternion (180 deg about X): (-1, 0, 0, 0)
static inline geometry_msgs::msg::Quaternion zDownQuat() {
  geometry_msgs::msg::Quaternion q;
  q.x = -1.0;
  q.y = 0.0;
  q.z = 0.0;
  q.w = 0.0;
  return q;
}

class ObjectManipulation {
public:
  using MoveGroupInterface = moveit::planning_interface::MoveGroupInterface;
  using Plan = MoveGroupInterface::Plan;
  using RobotTrajectory = moveit_msgs::msg::RobotTrajectory;
  using DetectedObject = custom_msgs::msg::DetectedObjects;

  explicit ObjectManipulation(const rclcpp::Node::SharedPtr &base_node)
      : base_node_(base_node) {
    RCLCPP_INFO(LOGGER, "[INIT] Initializing ObjectManipulation...");

    // Node options: allow MoveIt params to be injected from launch
    rclcpp::NodeOptions node_options;
    node_options.automatically_declare_parameters_from_overrides(true);

    // Dedicated node for MoveGroupInterface (best practice)
    move_group_node_ =
        rclcpp::Node::make_shared("move_group_node", node_options);

    // Use sim time (safe pattern to avoid double-declare)
    ensure_sim_time_true(move_group_node_);
    if (base_node_)
      ensure_sim_time_true(base_node_);

    // Wait until ROS time becomes active (Gazebo / sim)
    wait_ros_time_active(move_group_node_);

    // Spin move_group_node_ in its own executor thread
    executor_.add_node(move_group_node_);
    executor_thread_ = std::thread([this]() { executor_.spin(); });

    // Create MoveIt interfaces
    arm_ = std::make_shared<MoveGroupInterface>(move_group_node_,
                                                PLANNING_GROUP_ROBOT);
    gripper_ = std::make_shared<MoveGroupInterface>(move_group_node_,
                                                    PLANNING_GROUP_GRIPPER);

    // Start state monitors
    arm_->startStateMonitor();
    gripper_->startStateMonitor();

    // Wait for joint states to arrive
    wait_for_current_state(arm_, 5);

    // Print info
    RCLCPP_INFO(LOGGER, "Planning Frame: %s", arm_->getPlanningFrame().c_str());
    RCLCPP_INFO(LOGGER, "End Effector Link: %s",
                arm_->getEndEffectorLink().c_str());
    for (const auto &name : arm_->getJointModelGroupNames()) {
      RCLCPP_INFO(LOGGER, "Available group: %s", name.c_str());
    }

    // Planning tuning
    arm_->setPlanningTime(5.0);
    arm_->setNumPlanningAttempts(20);
    arm_->setGoalPositionTolerance(0.005);
    arm_->setGoalOrientationTolerance(0.05);
    arm_->setMaxVelocityScalingFactor(0.3);
    arm_->setMaxAccelerationScalingFactor(0.3);

    gripper_->setGoalTolerance(0.0001);
    gripper_->setMaxVelocityScalingFactor(0.05);
    gripper_->setMaxAccelerationScalingFactor(0.05);

    arm_->setPoseReferenceFrame(REF_FRAME);

    // Subscribe to cupholder detection and “gate” execution on it
    cupholder_promise_ = std::make_shared<std::promise<DetectedObject>>();
    cupholder_future_ = cupholder_promise_->get_future();

    cupholder_sub_ = move_group_node_->create_subscription<DetectedObject>(
        CUPHOLDER_TOPIC, 10,
        std::bind(&ObjectManipulation::cupholder_callback, this,
                  std::placeholders::_1));

    RCLCPP_INFO(LOGGER, "[INIT] Subscribed to %s (custom_msgs/DetectedObjects)",
                CUPHOLDER_TOPIC.c_str());
    RCLCPP_INFO(LOGGER, "[INIT] ObjectManipulation ready.");
  }

  ~ObjectManipulation() {
    executor_.cancel();
    if (executor_thread_.joinable())
      executor_thread_.join();
    RCLCPP_INFO(LOGGER, "[EXIT] ObjectManipulation shutdown.");
  }

  void execute() {
    using namespace std::chrono_literals;

    // ------------------------------------------------------------
    // 0) WAIT FOR CUPHOLDER DETECTION (but still place is fixed)
    // ------------------------------------------------------------
    RCLCPP_INFO(LOGGER,
                "[WAIT] Waiting for first cupholder detection on %s ...",
                CUPHOLDER_TOPIC.c_str());
    while (rclcpp::ok()) {
      if (cupholder_future_.wait_for(5s) == std::future_status::ready)
        break;
      RCLCPP_WARN(LOGGER,
                  "[WAIT] No cupholder detection yet, waiting another 5s...");
    }

    if (!rclcpp::ok())
      return;

    // We only use this to confirm perception works right now.
    const auto holder = cupholder_future_.get();
    RCLCPP_INFO(LOGGER,
                "[WAIT] Got cupholder detection id=%d at (%.3f, %.3f, %.3f). "
                "Using FIXED place pose for first test.",
                holder.object_id, holder.position.x, holder.position.y,
                holder.position.z);

    // ------------------------------------------------------------
    // 1) PICK from fixed cup pose
    // ------------------------------------------------------------
    const double cup_x = FIXED_CUP_X;
    const double cup_y = FIXED_CUP_Y;
    const double cup_z = FIXED_CUP_Z;

    const double pre_x = cup_x;
    const double pre_y = cup_y;
    const double pre_z = cup_z + PREGRASP_Z_OFFSET;

    RCLCPP_INFO(LOGGER, "[RUN] Starting sequence (fixed cup + fixed place, "
                        "with cupholder gating).");

    // Step 1: Home
    RCLCPP_INFO(LOGGER, "[1] Go to named pose: home");
    arm_->setStartStateToCurrentState();
    arm_->setNamedTarget("home");
    plan_and_execute_arm("[1] home");

    // Step 2: Open gripper
    RCLCPP_INFO(LOGGER, "[2] Open gripper");
    set_gripper_named("open");
    plan_and_execute_gripper("[2] open");

    // Step 3: Pregrasp above cup (Z-down)
    RCLCPP_INFO(LOGGER, "[3] Pregrasp above cup (%.3f, %.3f, %.3f)", pre_x,
                pre_y, pre_z);
    set_pose_target(pre_x, pre_y, pre_z, zDownQuat());
    plan_and_execute_arm("[3] pregrasp");

    // Step 4: Approach down (Cartesian)
    RCLCPP_INFO(LOGGER, "[4] Approach down %.3f m (Cartesian)",
                APPROACH_Z_DELTA);
    cartesian_delta(0.0, 0.0, APPROACH_Z_DELTA, "[4] approach");

    // Step 5: Close gripper
    RCLCPP_INFO(LOGGER, "[5] Close gripper");
    set_gripper_named("close");
    plan_and_execute_gripper("[5] close");

    std::this_thread::sleep_for(500ms);

    // Step 6: Retreat up (Cartesian)
    RCLCPP_INFO(LOGGER, "[6] Retreat up %.3f m (Cartesian)", RETREAT_Z_DELTA);
    cartesian_delta(0.0, 0.0, RETREAT_Z_DELTA, "[6] retreat");

    // ------------------------------------------------------------
    // 2) PLACE to fixed pose (first test)
    // ------------------------------------------------------------
    // Step 7: Move above place
    RCLCPP_INFO(LOGGER, "[7] Move above fixed place (%.3f, %.3f, %.3f)",
                FIXED_PLACE_X, FIXED_PLACE_Y, FIXED_PLACE_Z_PRE);
    set_pose_target(FIXED_PLACE_X, FIXED_PLACE_Y, FIXED_PLACE_Z_PRE,
                    zDownQuat());
    plan_and_execute_arm("[7] pre-place");

    // Step 8: Lower to drop pose
    RCLCPP_INFO(LOGGER, "[8] Lower to fixed drop (%.3f, %.3f, %.3f)",
                FIXED_PLACE_X, FIXED_PLACE_Y, FIXED_PLACE_Z_DROP);
    set_pose_target(FIXED_PLACE_X, FIXED_PLACE_Y, FIXED_PLACE_Z_DROP,
                    zDownQuat());
    plan_and_execute_arm("[8] drop");

    // Step 9: Open gripper (release)
    RCLCPP_INFO(LOGGER, "[9] Open gripper (release)");
    set_gripper_named("open");
    plan_and_execute_gripper("[9] release");

    // Step 10: Lift a bit after release (Cartesian)
    RCLCPP_INFO(LOGGER, "[10] Lift after release +0.20 m (Cartesian)");
    cartesian_delta(0.0, 0.0, +0.20, "[10] lift");

    // Step 11: Home
    RCLCPP_INFO(LOGGER, "[11] Go to named pose: home");
    arm_->setStartStateToCurrentState();
    arm_->setNamedTarget("home");
    plan_and_execute_arm("[11] home");

    RCLCPP_INFO(LOGGER, "[DONE] Sequence finished.");
  }

private:
  rclcpp::Node::SharedPtr base_node_;
  rclcpp::Node::SharedPtr move_group_node_;

  rclcpp::executors::SingleThreadedExecutor executor_;
  std::thread executor_thread_;

  std::shared_ptr<MoveGroupInterface> arm_;
  std::shared_ptr<MoveGroupInterface> gripper_;

  // Plans / trajectories
  Plan arm_plan_;
  Plan gripper_plan_;
  RobotTrajectory cart_traj_;

  // Cupholder detection gating
  std::shared_ptr<std::promise<DetectedObject>> cupholder_promise_;
  std::future<DetectedObject> cupholder_future_;
  rclcpp::Subscription<DetectedObject>::SharedPtr cupholder_sub_;
  bool cupholder_received_{false};

  // -------- Helpers --------
  static void ensure_sim_time_true(const rclcpp::Node::SharedPtr &node) {
    try {
      if (!node->has_parameter("use_sim_time")) {
        node->declare_parameter<bool>("use_sim_time", true);
      }
    } catch (const rclcpp::exceptions::ParameterAlreadyDeclaredException &) {
      // ignore
    }
    node->set_parameter(rclcpp::Parameter("use_sim_time", true));
  }

  static void wait_ros_time_active(const rclcpp::Node::SharedPtr &node) {
    using namespace std::chrono_literals;
    RCLCPP_INFO(LOGGER, "[TIME] Waiting for ROS time to become active...");
    auto start = std::chrono::steady_clock::now();
    while (!node->get_clock()->ros_time_is_active()) {
      if (std::chrono::steady_clock::now() - start > std::chrono::seconds(5)) {
        RCLCPP_WARN(LOGGER, "[TIME] ROS time not active after 5s. Continuing.");
        break;
      }
      std::this_thread::sleep_for(50ms);
    }
    RCLCPP_INFO(LOGGER, "[TIME] ros_time_is_active=%s",
                node->get_clock()->ros_time_is_active() ? "true" : "false");
  }

  static void
  wait_for_current_state(const std::shared_ptr<MoveGroupInterface> &arm,
                         int timeout_sec) {
    using namespace std::chrono_literals;
    RCLCPP_INFO(LOGGER, "[INIT] Waiting for /joint_states (up to %ds)...",
                timeout_sec);
    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);
    while (std::chrono::steady_clock::now() < deadline) {
      auto st = arm->getCurrentState(1.0);
      if (st) {
        RCLCPP_INFO(LOGGER, "[INIT] Current state received.");
        return;
      }
      RCLCPP_WARN(LOGGER, "[INIT] Still waiting for /joint_states...");
      std::this_thread::sleep_for(100ms);
    }
    RCLCPP_WARN(LOGGER, "[INIT] Proceeding without confirmed current state.");
  }

  void cupholder_callback(const DetectedObject::SharedPtr msg) {
    if (cupholder_received_)
      return;
    cupholder_received_ = true;

    try {
      cupholder_promise_->set_value(*msg);
    } catch (...) {
      // ignore if already set
    }

    cupholder_sub_.reset(); // unsubscribe after first detection

    RCLCPP_INFO(
        LOGGER,
        "[CB] cupholder id=%d pos=(%.3f, %.3f, %.3f) w=%.3f h=%.3f t=%.3f",
        msg->object_id, msg->position.x, msg->position.y, msg->position.z,
        msg->width, msg->height, msg->thickness);
  }

  void set_pose_target(double x, double y, double z,
                       const geometry_msgs::msg::Quaternion &q) {
    geometry_msgs::msg::Pose target;
    target.position.x = x;
    target.position.y = y;
    target.position.z = z;
    target.orientation = q;

    arm_->clearPoseTargets();
    arm_->setStartStateToCurrentState();
    arm_->setPoseTarget(target);
  }

  void set_gripper_named(const std::string &name) {
    gripper_->setStartStateToCurrentState();
    gripper_->setNamedTarget(name);
  }

  void plan_and_execute_arm(const std::string &tag) {
    auto ok = (arm_->plan(arm_plan_) == moveit::core::MoveItErrorCode::SUCCESS);
    if (!ok) {
      RCLCPP_ERROR(LOGGER, "%s: ARM plan FAILED", tag.c_str());
      return;
    }

    auto code = arm_->execute(arm_plan_);
    if (code != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(LOGGER, "%s: ARM execute FAILED (code=%d)", tag.c_str(),
                   code.val);
      return;
    }

    RCLCPP_INFO(LOGGER, "%s: ARM execute OK", tag.c_str());
  }

  void plan_and_execute_gripper(const std::string &tag) {
    auto ok = (gripper_->plan(gripper_plan_) ==
               moveit::core::MoveItErrorCode::SUCCESS);
    if (!ok) {
      RCLCPP_ERROR(LOGGER, "%s: GRIPPER plan FAILED", tag.c_str());
      return;
    }

    auto code = gripper_->execute(gripper_plan_);
    if (code != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(LOGGER, "%s: GRIPPER execute FAILED (code=%d)", tag.c_str(),
                   code.val);
      return;
    }

    RCLCPP_INFO(LOGGER, "%s: GRIPPER execute OK", tag.c_str());
  }

  void cartesian_delta(double dx, double dy, double dz,
                       const std::string &tag) {
    std::vector<geometry_msgs::msg::Pose> waypoints;
    auto cur = arm_->getCurrentPose().pose;

    waypoints.push_back(cur);
    cur.position.x += dx;
    cur.position.y += dy;
    cur.position.z += dz;
    waypoints.push_back(cur);

    const double fraction = arm_->computeCartesianPath(
        waypoints, EEF_STEP, JUMP_THRESHOLD, cart_traj_);

    if (fraction < CARTESIAN_MIN_FRACTION) {
      RCLCPP_ERROR(LOGGER,
                   "%s: Cartesian path FAILED (fraction=%.2f, need>=%.2f)",
                   tag.c_str(), fraction, CARTESIAN_MIN_FRACTION);
      return;
    }

    auto code = arm_->execute(cart_traj_);
    if (code != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(LOGGER, "%s: Cartesian execute FAILED (code=%d)",
                   tag.c_str(), code.val);
      return;
    }

    RCLCPP_INFO(LOGGER, "%s: Cartesian execute OK (fraction=%.2f)", tag.c_str(),
                fraction);
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  auto base_node = std::make_shared<rclcpp::Node>("object_manipulation");
  ObjectManipulation app(base_node);
  app.execute();

  rclcpp::shutdown();
  return 0;
}
