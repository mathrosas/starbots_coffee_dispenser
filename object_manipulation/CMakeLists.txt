cmake_minimum_required(VERSION 3.8)
project(object_manipulation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

add_executable(object_manipulation src/object_manipulation.cpp)

ament_target_dependencies(object_manipulation
  rclcpp
  geometry_msgs
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
  moveit_msgs
  tf2_ros
  tf2_geometry_msgs
)

install(TARGETS object_manipulation
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
