import socket
import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

HOST, PORT = "127.0.0.1", 5005  # host machine IP from container


class SensorForwarder(Node):
    def __init__(self):
        super().__init__("sensor_forwarder")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.create_subscription(Imu, "/imu1", self.imu1_callback, 10)
        self.create_subscription(Imu, "/imu2", self.imu2_callback, 10)

    def imu1_callback(self, msg):
        self.forward("/imu1", msg)

    def imu2_callback(self, msg):
        self.forward("/imu2", msg)

    def forward(self, topic, msg):
        data = {
            "topic": topic,
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
        self.sock.sendto(json.dumps(data).encode(), (HOST, PORT))


def main():
    rclpy.init()
    node = SensorForwarder()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
