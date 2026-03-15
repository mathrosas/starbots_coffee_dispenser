#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>

namespace object_manipulation {

enum class MotionStage {
  kPregrasp,
  kApproachCup,
  kRetreatWithCup,
  kRotateToPlace,
  kPrePlace,
  kInsertCup,
  kPostPlaceRetreat,
  kReturnHome,
  kRecoverySafePose,
  kRecoveryRetreatSmallZ,
  kPutBackAboveCup,
  kPutBackDescend,
  kPutBackRetreat,
};

struct MotionPolicy {
  std::string pipeline_id;
  std::string planner_id;
  double planning_time_sec{5.0};
  int num_planning_attempts{5};
  double velocity_scaling{0.1};
  double acceleration_scaling{0.05};
  bool linear_motion{false};
  bool require_orientation_constraint{false};
  std::size_t min_trajectory_points{2};
  bool replan_on_start_state_drift{true};
};

struct MotionStageHash {
  std::size_t operator()(MotionStage stage) const noexcept {
    return static_cast<std::size_t>(stage);
  }
};

using MotionPolicyMap = std::unordered_map<MotionStage, MotionPolicy, MotionStageHash>;

const char *to_string(MotionStage stage);

MotionPolicyMap
build_default_motion_policies(const std::string &ompl_pipeline,
                              const std::string &pilz_pipeline,
                              const std::string &pilz_lin_planner,
                              const std::string &constrained_ompl_planner);

} // namespace object_manipulation

