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
find_package(pcl_conversions REQUIRED)
# We require at least common and filters components from PCL.
find_package(PCL REQUIRED COMPONENTS common filters)

# Include PCL headers and definitions
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Build the executable from the source file in src/
add_executable(pointcloud_filter src/pointcloud_filter.cpp)

# Link against ROS2 and PCL libraries
ament_target_dependencies(pointcloud_filter 
  rclcpp
  sensor_msgs
  pcl_conversions
)
target_link_libraries(pointcloud_filter ${PCL_LIBRARIES})

# Install the executable
install(TARGETS
  pointcloud_filter
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # Skip copyright and cpplint (adjust if needed)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

