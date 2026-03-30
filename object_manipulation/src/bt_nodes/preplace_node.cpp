#include <bt_nodes/preplace_node.hpp>

PrePlaceNode::PrePlaceNode(const std::string &name,
                           const BT::NodeConfig &config,
                           object_manipulation::BtApi *api)
    : BT::SyncActionNode(name, config), api_(api) {}

BT::PortsList PrePlaceNode::providedPorts() { return {}; }

BT::NodeStatus PrePlaceNode::tick() { return api_->bt_pre_place(); }
