#/usr/bin/env bash
echo "Waiting 10s..."
sleep 10
echo "Circle publisher starting..."
source /opt/ros/humble/setup.bash
ros2 topic pub -r 5 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.5}}" > /dev/null
