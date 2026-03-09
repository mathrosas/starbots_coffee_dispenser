cmake_minimum_required(VERSION 3.8)
project(object_manipulation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(behaviortree_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(interactive_markers REQUIRED)
find_package(geometric_shapes REQUIRED)
find_package(control_msgs REQUIRED)
find_package(custom_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_visual_tools REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(add_coffee_scene src/add_coffee_scene.cpp)
add_executable(object_manipulation src/object_manipulation.cpp)

ament_target_dependencies(add_coffee_scene
	ament_cmake
	rclcpp
	moveit_ros_planning_interface)

ament_target_dependencies(object_manipulation
    ament_cmake
    ament_index_cpp
    behaviortree_cpp
    rclcpp
    rclcpp_action
    moveit_core
    moveit_ros_planning_interface
    interactive_markers
    moveit_ros_planning
    control_msgs
    custom_msgs
    moveit_msgs
    std_msgs)

install(TARGETS object_manipulation add_coffee_scene
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch bt_config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
