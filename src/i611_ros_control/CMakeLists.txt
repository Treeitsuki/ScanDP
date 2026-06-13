cmake_minimum_required(VERSION 3.0.2)
project(i611_ros_control)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  hardware_interface
  controller_manager
  position_controllers
  trajectory_msgs
  control_msgs
  xmlrpcpp
)

catkin_package(
  INCLUDE_DIRS include
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(i611_hw_node
  src/i611_hw_node.cpp
  src/i611_hw_interface.cpp
)

target_link_libraries(i611_hw_node
  ${catkin_LIBRARIES}
)
