#!/usr/bin/env bash
echo "Waiting 10s..."
sleep 10
echo "Random walk publisher starting..."
source /opt/ros/humble/setup.bash

while true; do
  linear_x=$(echo "scale=2; 0.5 + $RANDOM / 32767" | bc)
  angular_z=$(echo "scale=2; (2 * $RANDOM / 32767) - 1" | bc)
  
  echo "Publishing: linear_x=$linear_x, angular_z=$angular_z"
  
  ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: $linear_x, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: $angular_z}}"
  
  sleep 5
done
