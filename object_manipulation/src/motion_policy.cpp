#include "object_manipulation/motion_policy.hpp"

namespace object_manipulation {

const char *to_string(MotionStage stage) {
  switch (stage) {
  case MotionStage::kPregrasp:
    return "pregrasp";
  case MotionStage::kApproachCup:
    return "approach_cup";
  case MotionStage::kRetreatWithCup:
    return "retreat_with_cup";
  case MotionStage::kRotateToPlace:
    return "rotate_to_place";
  case MotionStage::kPrePlace:
    return "pre_place";
  case MotionStage::kInsertCup:
    return "insert_cup";
  case MotionStage::kPostPlaceRetreat:
    return "post_place_retreat";
  case MotionStage::kReturnHome:
    return "return_home";
  case MotionStage::kRecoverySafePose:
    return "recovery_safe_pose";
  case MotionStage::kRecoveryRetreatSmallZ:
    return "recovery_retreat_small_z";
  case MotionStage::kPutBackAboveCup:
    return "put_back_above_cup";
  case MotionStage::kPutBackDescend:
    return "put_back_descend";
  case MotionStage::kPutBackRetreat:
    return "put_back_retreat";
  default:
    return "unknown";
  }
}

MotionPolicyMap
build_default_motion_policies(const std::string &ompl_pipeline,
                              const std::string &pilz_pipeline,
                              const std::string &pilz_lin_planner,
                              const std::string &constrained_ompl_planner) {
  MotionPolicy ompl_free;
  ompl_free.pipeline_id = ompl_pipeline;
  ompl_free.planner_id = ""; // use OMPL group default (BiTRRT if configured)
  ompl_free.planning_time_sec = 20.0;
  ompl_free.num_planning_attempts = 20;
  ompl_free.velocity_scaling = 0.1;
  ompl_free.acceleration_scaling = 0.05;
  ompl_free.linear_motion = false;
  ompl_free.require_orientation_constraint = false;
  ompl_free.min_trajectory_points = 2;
  ompl_free.replan_on_start_state_drift = true;

  MotionPolicy ompl_constrained = ompl_free;
  ompl_constrained.planner_id = constrained_ompl_planner;
  ompl_constrained.require_orientation_constraint = true;
  ompl_constrained.planning_time_sec = 12.0;
  ompl_constrained.num_planning_attempts = 12;

  MotionPolicy pilz_lin;
  pilz_lin.pipeline_id = pilz_pipeline;
  pilz_lin.planner_id = pilz_lin_planner;
  pilz_lin.planning_time_sec = 3.0;
  pilz_lin.num_planning_attempts = 1;
  pilz_lin.velocity_scaling = 0.1;
  pilz_lin.acceleration_scaling = 0.05;
  pilz_lin.linear_motion = true;
  pilz_lin.require_orientation_constraint = false;
  pilz_lin.min_trajectory_points = 2;
  pilz_lin.replan_on_start_state_drift = false;

  MotionPolicyMap policies;

  // Global arm motions.
  policies[MotionStage::kPregrasp] = ompl_free;
  policies[MotionStage::kRotateToPlace] = ompl_free;
  policies[MotionStage::kReturnHome] = ompl_free;
  policies[MotionStage::kRecoverySafePose] = ompl_free;
  policies[MotionStage::kPutBackAboveCup] = ompl_free;

  // Constrained precise pre-place.
  policies[MotionStage::kPrePlace] = ompl_constrained;

  // Deterministic straight-line tool motion.
  policies[MotionStage::kApproachCup] = pilz_lin;
  policies[MotionStage::kRetreatWithCup] = pilz_lin;
  policies[MotionStage::kInsertCup] = pilz_lin;
  policies[MotionStage::kPostPlaceRetreat] = pilz_lin;
  policies[MotionStage::kRecoveryRetreatSmallZ] = pilz_lin;
  policies[MotionStage::kPutBackDescend] = pilz_lin;
  policies[MotionStage::kPutBackRetreat] = pilz_lin;

  return policies;
}

} // namespace object_manipulation

