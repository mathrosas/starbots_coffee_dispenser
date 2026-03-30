#include <bt_nodes/pick_node.hpp>

PickNode::PickNode(const std::string &name, const BT::NodeConfig &config,
                   object_manipulation::BtApi *api)
    : BT::SyncActionNode(name, config), api_(api) {}

BT::PortsList PickNode::providedPorts() { return {}; }

BT::NodeStatus PickNode::tick() { return api_->bt_pick(); }
