#include <bt_nodes/prepick_node.hpp>

PrePickNode::PrePickNode(const std::string &name, const BT::NodeConfig &config,
                         object_manipulation::BtApi *api)
    : BT::SyncActionNode(name, config), api_(api) {}

BT::PortsList PrePickNode::providedPorts() { return {}; }

BT::NodeStatus PrePickNode::tick() { return api_->bt_pre_pick(); }
