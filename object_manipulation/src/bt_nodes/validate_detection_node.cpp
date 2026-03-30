#include <bt_nodes/validate_detection_node.hpp>

ValidateDetectionNode::ValidateDetectionNode(const std::string &name,
                                             const BT::NodeConfig &config,
                                             object_manipulation::BtApi *api)
    : BT::SyncActionNode(name, config), api_(api) {}

BT::PortsList ValidateDetectionNode::providedPorts() { return {}; }

BT::NodeStatus ValidateDetectionNode::tick() {
  return api_->bt_validate_detection();
}
