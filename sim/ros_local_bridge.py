import socket
import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix

HOST, PORT = "127.0.0.1", 5005  # host machine IP from container


class SensorForwarder(Node):
    def __init__(self):
        super().__init__("sensor_forwarder")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.create_subscription(Imu, "/imu", self.imu_callback, 10)
        self.create_subscription(NavSatFix, "/gps", self.gps_callback, 10)

        self.imu_data = None
        self.gps_data = None

    def imu_callback(self, msg):
        self.forward("/imu", msg)

    def gps_callback(self, msg):
        self.forward("/gps", msg)

    def forward(self, topic, msg):
        if topic == "/gps":
            data = {
                "topic": topic,
                "latitude": msg.latitude,
                "longitude": msg.longitude,
                "altitude": msg.altitude,

            }
        if topic == "/imu":
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
