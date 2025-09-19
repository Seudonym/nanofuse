from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription(
        [
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                output="screen",
                arguments=["/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist"],
            ),
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                output="screen",
                arguments=[
                    "/model/mobile_robot/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry"
                ],
            ),
            # IMUs
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                output="screen",
                arguments=["/imu1@sensor_msgs/msg/Imu@gz.msgs.IMU"],
            ),
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                output="screen",
                arguments=["/imu2@sensor_msgs/msg/Imu@gz.msgs.IMU"],
            ),
        ]
    )

    return ld
