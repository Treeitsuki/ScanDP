#!/bin/bash

# RealSense をバックグラウンドで起動
roslaunch realsense2_camera rs_aligned_depth.launch &
sleep 3

# RTAB-Map をフォアグラウンドで起動
roslaunch rtabmap_ros rtabmap.launch \
  rtabmap_args:="--delete_db_on_start" \
  depth_topic:=/camera/aligned_depth_to_color/image_raw \
  rgb_topic:=/camera/color/image_raw \
  camera_info_topic:=/camera/color/camera_info
