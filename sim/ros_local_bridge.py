import socket
import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix
from nav_msgs.msg import Odometry

HOST, PORT = "127.0.0.1", 5005  # host machine IP from container


class SensorForwarder(Node):
    def __init__(self):
        super().__init__("sensor_forwarder")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Subscriptions
        self.create_subscription(Imu, "/imu", self.imu_callback, 10)
        self.create_subscription(NavSatFix, "/gps", self.gps_callback, 10)
        # Subscribe to ground truth odometry - check both possible topic names
        self.create_subscription(Odometry, "/model/mobile_robot/odometry", self.odom_callback, 10)

    def imu_callback(self, msg):
        """Forward IMU data"""
        data = {
            "topic": "/imu",
            "orientation": [
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ],
            "angular_velocity": [
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ],
            "linear_acceleration": [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ],
        }
        self._send_data(data)

    def gps_callback(self, msg):
        """Forward GPS data"""
        data = {
            "topic": "/gps",
            "latitude": msg.latitude,
            "longitude": msg.longitude,
            "altitude": msg.altitude,
        }
        self._send_data(data)

    def odom_callback(self, msg):
        """Forward ground truth odometry data"""
        data = {
            "topic": "/odom",
            "position": [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ],
            "orientation": [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ],
            "linear_velocity": [
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
            ],
            "angular_velocity": [
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z,
            ],
        }
        self._send_data(data)

    def _send_data(self, data):
        """Send data to collection script via UDP"""
        try:
            self.sock.sendto(json.dumps(data).encode(), (HOST, PORT))
        except Exception as e:
            self.get_logger().error(f"Failed to send data: {str(e)}")


def main():
    rclpy.init()
    node = SensorForwarder()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
