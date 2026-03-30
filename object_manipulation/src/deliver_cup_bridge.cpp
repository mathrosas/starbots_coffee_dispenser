#include <custom_msgs/action/deliver_cup.hpp>
#include <custom_msgs/srv/pick_place_cup.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <std_msgs/msg/string.hpp>

#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

class PickPlaceCupBridge : public rclcpp::Node {
public:
  using DeliverCup = custom_msgs::action::DeliverCup;
  using PickPlaceCup = custom_msgs::srv::PickPlaceCup;
  using DeliverCupGoalHandle = rclcpp_action::ClientGoalHandle<DeliverCup>;

  PickPlaceCupBridge() : rclcpp::Node("deliver_cup_bridge") {
    // Service callback blocks until action result arrives, so action callbacks
    // must run in a different callback group to avoid deadlock.
    service_callback_group_ = this->create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    action_callback_group_ =
        this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    action_client_ = rclcpp_action::create_client<DeliverCup>(
        this, "/deliver_cup", action_callback_group_);
    status_pub_ =
        this->create_publisher<std_msgs::msg::String>("/robot_status_feedback",
                                                      10);
    service_server_ = this->create_service<PickPlaceCup>(
        "/deliver_cup",
        std::bind(&PickPlaceCupBridge::handle_service, this,
                  std::placeholders::_1, std::placeholders::_2),
        rmw_qos_profile_services_default, service_callback_group_);

    publish_status("Deliver cup bridge ready. Service: /deliver_cup");
    RCLCPP_INFO(this->get_logger(),
                "Deliver cup bridge started. Service=/deliver_cup "
                "Action=/deliver_cup");
  }

private:
  class ScopeExit {
  public:
    explicit ScopeExit(std::function<void()> fn) : fn_(std::move(fn)) {}
    ~ScopeExit() {
      if (fn_) {
        fn_();
      }
    }
    ScopeExit(const ScopeExit &) = delete;
    ScopeExit &operator=(const ScopeExit &) = delete;
    ScopeExit(ScopeExit &&) = delete;
    ScopeExit &operator=(ScopeExit &&) = delete;

  private:
    std::function<void()> fn_;
  };

  struct GoalState {
    std::mutex mutex;
    std::condition_variable cv;
    bool done{false};
    bool accepted{false};
    bool success{false};
    std::string message{"No action result available"};
  };

  void handle_service(const std::shared_ptr<PickPlaceCup::Request> request,
                      std::shared_ptr<PickPlaceCup::Response> response) {
    if (request->goal_cup_holder == 0U) {
      response->result = "Invalid request: goal_cup_holder must be > 0";
      publish_status(response->result);
      RCLCPP_WARN(this->get_logger(), "%s", response->result.c_str());
      return;
    }

    {
      std::lock_guard<std::mutex> lock(service_mutex_);
      if (service_in_progress_) {
        response->result =
            "Bridge busy: another /deliver_cup request is in progress";
        publish_status(response->result);
        RCLCPP_WARN(this->get_logger(), "%s", response->result.c_str());
        return;
      }
      service_in_progress_ = true;
    }

    ScopeExit release_guard([this]() {
      std::lock_guard<std::mutex> lock(service_mutex_);
      service_in_progress_ = false;
    });

    if (!action_client_->wait_for_action_server(std::chrono::seconds(5))) {
      response->result = "Action server /deliver_cup not available";
      publish_status(response->result);
      RCLCPP_ERROR(this->get_logger(), "%s", response->result.c_str());
      return;
    }

    auto goal = DeliverCup::Goal();
    goal.cupholder_id = static_cast<uint32_t>(request->goal_cup_holder);

    std::ostringstream start_msg;
    start_msg << "Coffee delivery started for cupholder_id="
              << static_cast<unsigned int>(request->goal_cup_holder);
    publish_status(start_msg.str());

    auto state = std::make_shared<GoalState>();

    rclcpp_action::Client<DeliverCup>::SendGoalOptions options;
    options.goal_response_callback =
        [this, state](std::shared_ptr<DeliverCupGoalHandle> goal_handle) {
          if (!goal_handle) {
            {
              std::lock_guard<std::mutex> lock(state->mutex);
              state->accepted = false;
              state->done = true;
              state->message = "Action rejected by /deliver_cup";
            }
            publish_status(state->message);
            state->cv.notify_one();
            return;
          }
          {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->accepted = true;
          }
          publish_status("Action accepted by /deliver_cup");
        };

    options.feedback_callback =
        [this](std::shared_ptr<DeliverCupGoalHandle> /*goal_handle*/,
               const std::shared_ptr<const DeliverCup::Feedback> feedback) {
          if (!feedback) {
            return;
          }
          std::ostringstream msg;
          msg << "stage=" << feedback->stage << " progress=" << feedback->progress
              << " cupholder_id="
              << static_cast<unsigned int>(feedback->cupholder_id);
          publish_status(msg.str());
        };

    options.result_callback =
        [this, state](const DeliverCupGoalHandle::WrappedResult &result) {
          {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->done = true;

            if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
              if (result.result) {
                state->success = result.result->success;
                state->message = result.result->message;
              } else {
                state->success = true;
                state->message = "DeliverCup succeeded";
              }
            } else if (result.code == rclcpp_action::ResultCode::ABORTED) {
              state->success = false;
              state->message = "DeliverCup aborted";
            } else if (result.code == rclcpp_action::ResultCode::CANCELED) {
              state->success = false;
              state->message = "DeliverCup canceled";
            } else {
              state->success = false;
              state->message = "DeliverCup returned unknown result";
            }
          }
          publish_status(state->message);
          state->cv.notify_one();
        };

    action_client_->async_send_goal(goal, options);

    std::unique_lock<std::mutex> lock(state->mutex);
    const bool completed = state->cv.wait_for(
        lock, std::chrono::minutes(10), [&state]() { return state->done; });
    if (!completed) {
      lock.unlock();
      response->result = "Timed out waiting for /deliver_cup result";
      publish_status(response->result);
      RCLCPP_ERROR(this->get_logger(), "%s", response->result.c_str());
      return;
    }

    if (!state->accepted && state->message.empty()) {
      state->message = "Action goal rejected";
    }
    response->result = state->message;
    RCLCPP_INFO(this->get_logger(), "Service response: %s",
                response->result.c_str());
  }

  void publish_status(const std::string &text) {
    std_msgs::msg::String msg;
    msg.data = text;
    status_pub_->publish(msg);
  }

  rclcpp_action::Client<DeliverCup>::SharedPtr action_client_;
  rclcpp::Service<PickPlaceCup>::SharedPtr service_server_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::CallbackGroup::SharedPtr service_callback_group_;
  rclcpp::CallbackGroup::SharedPtr action_callback_group_;
  std::mutex service_mutex_;
  bool service_in_progress_{false};
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PickPlaceCupBridge>();
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(),
                                                     2);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
