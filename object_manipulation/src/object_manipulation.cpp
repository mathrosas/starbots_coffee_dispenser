#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

#include <custom_msgs/msg/detected_objects.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>

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

// Cupholder detection
static const std::string CUPHOLDER_TOPIC = "/cup_holder_detected";

// Frames (your python publishes positions in base_link)
static const std::string REF_FRAME = "base_link";

// -------- Fixed cup pose (pick test) --------
static constexpr double FIXED_CUP_X = 0.298;
static constexpr double FIXED_CUP_Y = 0.330;
static constexpr double FIXED_CUP_Z = 0.035;

// -------- Motion tuning --------
static constexpr double PREGRASP_Z_OFFSET = 0.30; // above cup
static constexpr double PICK_APPROACH_Z_DELTA = -0.12;
static constexpr double PICK_RETREAT_Z_DELTA = +0.30;

// Cartesian planning
static constexpr double EEF_STEP = 0.01;
static constexpr double JUMP_THRESHOLD = 0.0;
static constexpr double CARTESIAN_MIN_FRACTION = 0.50;

// Step rotate delta (your logic)
static constexpr double SHOULDER_PAN_DELTA_RAD = 2.0 * M_PI / 3.0; // +120deg

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
  using DetectedObjectsMsg = custom_msgs::msg::DetectedObjects;

  explicit ObjectManipulation(const rclcpp::Node::SharedPtr &base_node)
      : base_node_(base_node) {
    RCLCPP_INFO(LOGGER, "[INIT] Initializing ObjectManipulation...");

    rclcpp::NodeOptions node_options;
    node_options.automatically_declare_parameters_from_overrides(true);

    move_group_node_ =
        rclcpp::Node::make_shared("move_group_node", node_options);

    // Always force sim time for this node (still allow overriding via ros-args)
    ensure_sim_time_true(move_group_node_);
    if (base_node_)
      ensure_sim_time_true(base_node_);

    // ---- Placement tuning parameters (runtime) ----
    // Guard declarations because ros-args overrides may auto-declare them.
    if (!move_group_node_->has_parameter("holder_dx"))
      move_group_node_->declare_parameter<double>("holder_dx", 0.0);
    if (!move_group_node_->has_parameter("holder_dy"))
      move_group_node_->declare_parameter<double>("holder_dy", 0.0);
    if (!move_group_node_->has_parameter("holder_dz"))
      move_group_node_->declare_parameter<double>("holder_dz", 0.0);

    if (!move_group_node_->has_parameter("cup_dx"))
      move_group_node_->declare_parameter<double>("cup_dx", 0.0);
    if (!move_group_node_->has_parameter("cup_dy"))
      move_group_node_->declare_parameter<double>("cup_dy", 0.0);
    if (!move_group_node_->has_parameter("cup_dz"))
      move_group_node_->declare_parameter<double>("cup_dz", 0.0);

    if (!move_group_node_->has_parameter("place_hover_z_offset"))
      move_group_node_->declare_parameter<double>("place_hover_z_offset", 0.20);
    if (!move_group_node_->has_parameter("place_insert_z_delta"))
      move_group_node_->declare_parameter<double>("place_insert_z_delta",
                                                  -0.23);
    if (!move_group_node_->has_parameter("place_hover_min_z"))
      move_group_node_->declare_parameter<double>("place_hover_min_z", -0.30);
    if (!move_group_node_->has_parameter("adaptive_insert_to_place"))
      move_group_node_->declare_parameter<bool>("adaptive_insert_to_place",
                                                true);

    // Which cupholder to use (Option: -p ch:=ch_1)
    if (!move_group_node_->has_parameter("ch"))
      move_group_node_->declare_parameter<std::string>("ch", "ch_1");

    // Repetition support (optional): -p repeat:=3 (default 1)
    if (!move_group_node_->has_parameter("repeat"))
      move_group_node_->declare_parameter<int>("repeat", 1);

    // Parse target holder from param
    const auto ch = move_group_node_->get_parameter("ch").as_string();
    target_holder_id_ = parse_ch_to_id(ch);

    RCLCPP_INFO(LOGGER, "[INIT] Target cupholder param ch='%s' -> id=%d",
                ch.c_str(), target_holder_id_);

    wait_ros_time_active(move_group_node_);

    executor_.add_node(move_group_node_);
    executor_thread_ = std::thread([this]() { executor_.spin(); });

    arm_ = std::make_shared<MoveGroupInterface>(move_group_node_,
                                                PLANNING_GROUP_ROBOT);
    gripper_ = std::make_shared<MoveGroupInterface>(move_group_node_,
                                                    PLANNING_GROUP_GRIPPER);

    arm_->startStateMonitor();
    gripper_->startStateMonitor();

    wait_for_current_state(arm_, 5);

    RCLCPP_INFO(LOGGER, "Planning Frame: %s", arm_->getPlanningFrame().c_str());
    RCLCPP_INFO(LOGGER, "End Effector Link: %s",
                arm_->getEndEffectorLink().c_str());

    arm_->setPlanningTime(60.0);
    arm_->setNumPlanningAttempts(20);
    arm_->setGoalPositionTolerance(0.005);
    arm_->setGoalOrientationTolerance(0.05);
    arm_->setMaxVelocityScalingFactor(0.3);
    arm_->setMaxAccelerationScalingFactor(0.3);

    gripper_->setGoalTolerance(0.0001);
    gripper_->setMaxVelocityScalingFactor(0.05);
    gripper_->setMaxAccelerationScalingFactor(0.05);

    arm_->setPoseReferenceFrame(REF_FRAME);

    // Subscribe to cupholder detections; we will filter by target_holder_id_
    cupholder_sub_ = move_group_node_->create_subscription<DetectedObjectsMsg>(
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
    // 0) WAIT FOR TARGET CUPHOLDER DETECTION
    // ------------------------------------------------------------
    RCLCPP_INFO(LOGGER, "[WAIT] Waiting for target cupholder id=%d on %s ...",
                target_holder_id_, CUPHOLDER_TOPIC.c_str());

    while (rclcpp::ok() && !have_latest_holder_) {
      std::this_thread::sleep_for(100ms);
    }
    if (!rclcpp::ok())
      return;

    // Read tuning params
    const double holder_dx =
        move_group_node_->get_parameter("holder_dx").as_double();
    const double holder_dy =
        move_group_node_->get_parameter("holder_dy").as_double();
    const double holder_dz =
        move_group_node_->get_parameter("holder_dz").as_double();

    const double cup_dx = move_group_node_->get_parameter("cup_dx").as_double();
    const double cup_dy = move_group_node_->get_parameter("cup_dy").as_double();
    const double cup_dz = move_group_node_->get_parameter("cup_dz").as_double();

    const double place_hover_z_offset =
        move_group_node_->get_parameter("place_hover_z_offset").as_double();
    const double place_insert_z_delta =
        move_group_node_->get_parameter("place_insert_z_delta").as_double();
    const double place_hover_min_z =
        move_group_node_->get_parameter("place_hover_min_z").as_double();
    const bool adaptive_insert_to_place =
        move_group_node_->get_parameter("adaptive_insert_to_place").as_bool();

    int repeat = move_group_node_->get_parameter("repeat").as_int();
    if (repeat < 1)
      repeat = 1;

    RCLCPP_INFO(
        LOGGER,
        "[TUNE] holder_d=(%.3f, %.3f, %.3f) cup_d=(%.3f, %.3f, %.3f) "
        "hover_z=%.3f insert_dz=%.3f min_hover_z=%.3f adaptive_insert=%s "
        "repeat=%d",
        holder_dx, holder_dy, holder_dz, cup_dx, cup_dy, cup_dz,
        place_hover_z_offset, place_insert_z_delta, place_hover_min_z,
        adaptive_insert_to_place ? "true" : "false", repeat);

    // Snapshot holder pose at start (you can also keep updating per loop if you
    // want)
    DetectedObjectsMsg holder = latest_holder_;

    RCLCPP_INFO(LOGGER,
                "[TARGET] Using holder id=%d at (%.3f, %.3f, %.3f) w=%.3f "
                "h=%.3f t=%.3f",
                holder.object_id, holder.position.x, holder.position.y,
                holder.position.z, holder.width, holder.height,
                holder.thickness);

    // ------------------------------------------------------------
    // 1) PICK from fixed cup pose (your existing test)
    // ------------------------------------------------------------
    const double cup_x = FIXED_CUP_X;
    const double cup_y = FIXED_CUP_Y;
    const double cup_z = FIXED_CUP_Z;

    const double pre_x = cup_x;
    const double pre_y = cup_y;
    const double pre_z = cup_z + PREGRASP_Z_OFFSET;

    RCLCPP_INFO(LOGGER, "[RUN] Starting sequence.");

    // Step 1: Home
    RCLCPP_INFO(LOGGER, "[1] Go to named pose: home");
    arm_->setStartStateToCurrentState();
    arm_->setNamedTarget("home");
    if (!plan_and_execute_arm("[1] home"))
      return;

    // Step 2: Open gripper
    RCLCPP_INFO(LOGGER, "[2] Open gripper");
    set_gripper_named("open");
    if (!plan_and_execute_gripper("[2] open"))
      return;

    // Step 3: Pregrasp above cup (Z-down)
    RCLCPP_INFO(LOGGER, "[3] Pregrasp above cup (%.3f, %.3f, %.3f)", pre_x,
                pre_y, pre_z);
    set_pose_target(pre_x, pre_y, pre_z, zDownQuat());
    if (!plan_and_execute_arm("[3] pregrasp"))
      return;

    // Step 4: Approach down (Cartesian)
    RCLCPP_INFO(LOGGER, "[4] Approach down %.3f m (Cartesian)",
                PICK_APPROACH_Z_DELTA);
    if (!cartesian_delta(0.0, 0.0, PICK_APPROACH_Z_DELTA, "[4] approach_pick"))
      return;

    // Step 5: Close gripper
    RCLCPP_INFO(LOGGER, "[5] Close gripper");
    set_gripper_named("close");
    if (!plan_and_execute_gripper("[5] close"))
      return;

    std::this_thread::sleep_for(500ms);

    // Step 6: Retreat up (Cartesian)
    RCLCPP_INFO(LOGGER, "[6] Retreat up %.3f m (Cartesian)",
                PICK_RETREAT_Z_DELTA);
    if (!cartesian_delta(0.0, 0.0, PICK_RETREAT_Z_DELTA, "[6] retreat_pick"))
      return;

    // Step 7: Rotate shoulder_pan_joint (your existing logic)
    RCLCPP_INFO(LOGGER, "[7] Rotate shoulder_pan_joint by +%.1f deg",
                SHOULDER_PAN_DELTA_RAD * 180.0 / M_PI);
    if (!rotate_shoulder_pan_relative(SHOULDER_PAN_DELTA_RAD,
                                      "[7] rotate_shoulder_pan"))
      return;

    std::this_thread::sleep_for(500ms);
    arm_->setStartStateToCurrentState();

    // ------------------------------------------------------------
    // 2) PLACE INTO CUPHOLDER (KURALME-STYLE), possibly repeated
    // ------------------------------------------------------------

    for (int k = 1; k <= repeat && rclcpp::ok(); ++k) {
      // If you want to re-acquire the latest holder pose each time, uncomment:
      // holder = latest_holder_;

      // Build the “hole center” target in base_link, with calibration
      const double hole_x = holder.position.x + holder_dx;
      const double hole_y = holder.position.y + holder_dy;
      const double hole_z = holder.position.z + holder_dz;

      const double place_x = hole_x + cup_dx;
      const double place_y = hole_y + cup_dy;
      const double place_z = hole_z + cup_dz;

      const double hover_x = place_x;
      const double hover_y = place_y;
      double hover_z = place_z + place_hover_z_offset;
      if (hover_z < place_hover_min_z) {
        RCLCPP_WARN(LOGGER, "[PLACE %d/%d] hover_z clamped from %.3f to %.3f",
                    k, repeat, hover_z, place_hover_min_z);
        hover_z = place_hover_min_z;
      }

      double insert_dz = place_insert_z_delta;
      if (adaptive_insert_to_place) {
        insert_dz = place_z - hover_z;
      }

      RCLCPP_INFO(LOGGER,
                  "[PLACE %d/%d] holder id=%d hover=(%.3f, %.3f, %.3f) "
                  "place=(%.3f, %.3f, %.3f) insert_dz=%.3f",
                  k, repeat, holder.object_id, hover_x, hover_y, hover_z,
                  place_x, place_y, place_z, insert_dz);

      // Step 8: Move to hover above detected holder (Z-down)
      set_pose_target(hover_x, hover_y, hover_z, zDownQuat());
      if (!plan_and_execute_arm("[8] hover_holder")) {
        RCLCPP_ERROR(LOGGER, "[PLACE %d/%d] Aborting: hover plan failed.", k,
                     repeat);
        return;
      }

      // Step 9: Insert straight down by fixed dz (kuralme style)
      if (!cartesian_delta(0.0, 0.0, insert_dz, "[9] insert")) {
        RCLCPP_ERROR(LOGGER, "[PLACE %d/%d] Aborting: insert motion failed.", k,
                     repeat);
        return;
      }

      // Step 10: Open gripper (release) ONLY on first placement attempt
      // If you want to keep the cup after first attempt, remove this condition.
      if (k == 1) {
        RCLCPP_INFO(LOGGER, "[10] Open gripper (release)");
        set_gripper_named("open");
        if (!plan_and_execute_gripper("[10] release")) {
          RCLCPP_ERROR(LOGGER, "[PLACE %d/%d] Aborting: release failed.", k,
                       repeat);
          return;
        }
        std::this_thread::sleep_for(500ms);
        arm_->setStartStateToCurrentState();
      }

      // Step 11: Retreat up (undo insertion)
      if (!cartesian_delta(0.0, 0.0, -insert_dz, "[11] retreat_place")) {
        RCLCPP_ERROR(LOGGER,
                     "[PLACE %d/%d] Aborting: retreat-after-place failed.", k,
                     repeat);
        return;
      }
    }

    // Step 12: Home
    RCLCPP_INFO(LOGGER, "[12] Go to named pose: home");
    arm_->setStartStateToCurrentState();
    arm_->setNamedTarget("home");
    if (!plan_and_execute_arm("[12] home"))
      return;

    RCLCPP_INFO(LOGGER, "[DONE] Sequence finished.");
  }

private:
  rclcpp::Node::SharedPtr base_node_;
  rclcpp::Node::SharedPtr move_group_node_;

  rclcpp::executors::SingleThreadedExecutor executor_;
  std::thread executor_thread_;

  std::shared_ptr<MoveGroupInterface> arm_;
  std::shared_ptr<MoveGroupInterface> gripper_;

  Plan arm_plan_;
  Plan gripper_plan_;
  RobotTrajectory cart_traj_;

  rclcpp::Subscription<DetectedObjectsMsg>::SharedPtr cupholder_sub_;

  // Target selection + latest holder cache
  int target_holder_id_{1};
  DetectedObjectsMsg latest_holder_;
  bool have_latest_holder_{false};

  // -------- Helpers --------
  static int parse_ch_to_id(const std::string &s) {
    // Accept "ch_1", "ch_2", ... ; fallback to 1
    if (s.rfind("ch_", 0) != 0)
      return 1;
    try {
      int id = std::stoi(s.substr(3));
      return (id >= 1) ? id : 1;
    } catch (...) {
      return 1;
    }
  }

  static void ensure_sim_time_true(const rclcpp::Node::SharedPtr &node) {
    try {
      if (!node->has_parameter("use_sim_time")) {
        node->declare_parameter<bool>("use_sim_time", true);
      }
    } catch (const rclcpp::exceptions::ParameterAlreadyDeclaredException &) {
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

  void cupholder_callback(const DetectedObjectsMsg::SharedPtr msg) {
    // Only accept the chosen holder id
    if (msg->object_id != target_holder_id_) {
      RCLCPP_DEBUG(LOGGER, "[CB] Ignoring id=%d (want id=%d)", msg->object_id,
                   target_holder_id_);
      return;
    }

    latest_holder_ = *msg;
    have_latest_holder_ = true;

    RCLCPP_INFO(LOGGER,
                "[CB] Updated target holder id=%d pos=(%.3f, %.3f, %.3f) "
                "w=%.3f h=%.3f t=%.3f",
                msg->object_id, msg->position.x, msg->position.y,
                msg->position.z, msg->width, msg->height, msg->thickness);
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

  bool plan_and_execute_arm(const std::string &tag) {
    auto ok = (arm_->plan(arm_plan_) == moveit::core::MoveItErrorCode::SUCCESS);
    if (!ok) {
      RCLCPP_ERROR(LOGGER, "%s: ARM plan FAILED", tag.c_str());
      return false;
    }
    auto code = arm_->execute(arm_plan_);
    if (code != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(LOGGER, "%s: ARM execute FAILED (code=%d)", tag.c_str(),
                   code.val);
      return false;
    }
    RCLCPP_INFO(LOGGER, "%s: ARM execute OK", tag.c_str());
    return true;
  }

  bool plan_and_execute_gripper(const std::string &tag) {
    auto ok = (gripper_->plan(gripper_plan_) ==
               moveit::core::MoveItErrorCode::SUCCESS);
    if (!ok) {
      RCLCPP_ERROR(LOGGER, "%s: GRIPPER plan FAILED", tag.c_str());
      return false;
    }
    auto code = gripper_->execute(gripper_plan_);
    if (code != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(LOGGER, "%s: GRIPPER execute FAILED (code=%d)", tag.c_str(),
                   code.val);
      return false;
    }
    RCLCPP_INFO(LOGGER, "%s: GRIPPER execute OK", tag.c_str());
    return true;
  }

  bool cartesian_delta(double dx, double dy, double dz,
                       const std::string &tag) {
    std::vector<geometry_msgs::msg::Pose> waypoints;
    auto cur = arm_->getCurrentPose().pose;

    waypoints.push_back(cur);
    cur.position.x += dx;
    cur.position.y += dy;
    cur.position.z += dz;
    waypoints.push_back(cur);

    const double fraction = arm_->computeCartesianPath(
        waypoints, EEF_STEP, JUMP_THRESHOLD, cart_traj_, true);

    if (fraction < CARTESIAN_MIN_FRACTION) {
      RCLCPP_ERROR(LOGGER,
                   "%s: Cartesian path FAILED (fraction=%.2f, need>=%.2f)",
                   tag.c_str(), fraction, CARTESIAN_MIN_FRACTION);
      return false;
    }

    auto code = arm_->execute(cart_traj_);
    if (code != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(LOGGER, "%s: Cartesian execute FAILED (code=%d)",
                   tag.c_str(), code.val);
      return false;
    }

    RCLCPP_INFO(LOGGER, "%s: Cartesian execute OK (fraction=%.2f)", tag.c_str(),
                fraction);
    return true;
  }

  bool rotate_shoulder_pan_relative(double delta_rad, const std::string &tag) {
    auto state = arm_->getCurrentState(1.0);
    if (!state) {
      RCLCPP_ERROR(LOGGER, "%s: could not get current state", tag.c_str());
      return false;
    }

    const moveit::core::JointModelGroup *jmg =
        state->getJointModelGroup(PLANNING_GROUP_ROBOT);
    if (!jmg) {
      RCLCPP_ERROR(LOGGER, "%s: no JointModelGroup for %s", tag.c_str(),
                   PLANNING_GROUP_ROBOT.c_str());
      return false;
    }

    std::vector<double> joints;
    state->copyJointGroupPositions(jmg, joints);
    const auto &names = jmg->getVariableNames();

    int idx = -1;
    for (size_t i = 0; i < names.size(); ++i) {
      if (names[i] == "shoulder_pan_joint") {
        idx = static_cast<int>(i);
        break;
      }
    }
    if (idx < 0 || static_cast<size_t>(idx) >= joints.size()) {
      RCLCPP_ERROR(LOGGER, "%s: shoulder_pan_joint not found in group",
                   tag.c_str());
      return false;
    }

    joints[idx] += delta_rad;
    while (joints[idx] > M_PI)
      joints[idx] -= 2.0 * M_PI;
    while (joints[idx] < -M_PI)
      joints[idx] += 2.0 * M_PI;

    arm_->setStartStateToCurrentState();
    arm_->setJointValueTarget(joints);

    Plan plan;
    auto ok = (arm_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
    if (!ok) {
      RCLCPP_WARN(LOGGER, "%s: plan FAILED", tag.c_str());
      return false;
    }

    auto code = arm_->execute(plan);
    if (code != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(LOGGER, "%s: execute FAILED (code=%d)", tag.c_str(),
                   code.val);
      return false;
    }
    RCLCPP_INFO(LOGGER, "%s: execute OK", tag.c_str());
    return true;
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
