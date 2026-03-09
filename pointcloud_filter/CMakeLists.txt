cmake_minimum_required(VERSION 3.8)
project(pointcloud_filter)

# Compiler warnings
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

# Build the executable from the source file in src/
add_executable(pointcloud_filter_node src/pointcloud_filter.cpp)

# Link against ROS2 libraries
ament_target_dependencies(pointcloud_filter_node 
  rclcpp
  sensor_msgs
)

# Install the executable
install(TARGETS
  pointcloud_filter_node
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # Skip copyright and cpplint (adjust if needed)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
