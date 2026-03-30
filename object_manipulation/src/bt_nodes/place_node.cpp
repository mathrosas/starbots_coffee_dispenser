#include <bt_nodes/place_node.hpp>

PlaceNode::PlaceNode(const std::string &name, const BT::NodeConfig &config,
                     object_manipulation::BtApi *api)
    : BT::SyncActionNode(name, config), api_(api) {}

BT::PortsList PlaceNode::providedPorts() { return {}; }

BT::NodeStatus PlaceNode::tick() { return api_->bt_place(); }
