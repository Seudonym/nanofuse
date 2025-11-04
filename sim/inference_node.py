import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix
from nav_msgs.msg import Odometry
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import pandas as pd  # Removed - using numpy arrays directly for better performance
from collections import deque
import tf2_ros
from geometry_msgs.msg import TransformStamped
import joblib
import os


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class TransformerFusionModel(nn.Module):
    """Transformer model matching the training script"""
    def __init__(
        self,
        input_dim,
        output_dim,
        d_model=128,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        max_seq_len=200,
    ):
        super(TransformerFusionModel, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim_feedforward, dim_feedforward // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc_out = nn.Linear(dim_feedforward // 2, output_dim)
        
    def forward(self, src):
        x = self.input_projection(src)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc_out(x)
        return x


class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')
        
        # Model configuration (will be loaded from checkpoint)
        self.model_config = None
        self.window_size = 100
        
        # Load scalers and model configuration
        # With docker-compose mount: ..:/workdir/, the datasets folder is at /workdir/datasets/
        base_path = "/workdir"
        scaler_path = os.path.join(base_path, "datasets", "scalers.pkl")
        model_path = os.path.join(base_path, "datasets", "transformer_fusion_model.pth")
        
        if not os.path.exists(scaler_path):
            self.get_logger().error(f"Scalers not found at {scaler_path}")
            raise FileNotFoundError(f"Scalers not found at {scaler_path}")
        
        if not os.path.exists(model_path):
            self.get_logger().error(f"Model not found at {model_path}")
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load scalers
        try:
            scaler_data = joblib.load(scaler_path)
            self.X_scaler = scaler_data['X_scaler']
            self.y_scaler = scaler_data['y_scaler']
            self.input_features = scaler_data['input_features']
            self.output_labels = scaler_data['output_labels']
            self.gps_reference = scaler_data.get('gps_reference', None)
        except Exception as e:
            self.get_logger().error(f"Failed to load scalers: {str(e)}")
            raise
        
        self.get_logger().info(f"Loaded scalers from {scaler_path}")
        if self.gps_reference:
            self.get_logger().info(f"GPS reference: lat={self.gps_reference['lat']:.6f}, lon={self.gps_reference['lon']:.6f}")
        
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
        except Exception as e:
            self.get_logger().error(f"Failed to load model checkpoint: {str(e)}")
            raise
        
        # Get model configuration from checkpoint
        if 'model_config' in checkpoint:
            self.model_config = checkpoint['model_config']
        else:
            # Fallback to default if not in checkpoint
            self.model_config = {
                'input_dim': len(self.input_features),
                'output_dim': len(self.output_labels),
                'd_model': 128,
                'nhead': 8,
                'num_encoder_layers': 4,
                'dim_feedforward': 512,
                'dropout': 0.1,
            }
        
        self.model = TransformerFusionModel(
            input_dim=self.model_config['input_dim'],
            output_dim=self.model_config['output_dim'],
            d_model=self.model_config['d_model'],
            nhead=self.model_config['nhead'],
            num_encoder_layers=self.model_config['num_encoder_layers'],
            dim_feedforward=self.model_config['dim_feedforward'],
            dropout=self.model_config['dropout'],
            max_seq_len=self.window_size,
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.get_logger().info(f"Model loaded from {model_path} on device: {self.device}")
        
        # TF Broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Subscribers
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.gps_sub = self.create_subscription(NavSatFix, '/gps', self.gps_callback, 10)
        
        # Publisher
        self.fused_odom_pub = self.create_publisher(Odometry, '/fused_odom', 10)
        
        # Data buffers
        self.imu_data = deque()
        self.gps_data = deque()
        
        # Rate limiting for inference (prevent excessive inference calls)
        self.last_inference_time = 0.0
        self.min_inference_interval = 0.1  # Minimum 100ms between inferences (10 Hz max)
        
        # GPS synchronization parameters
        self.max_gps_time_diff = 1.0  # Maximum 1 second difference for GPS matching
        
    def convert_gps_to_local(self, lat, lon):
        """Convert GPS coordinates to local Cartesian coordinates (meters)"""
        if self.gps_reference is None:
            return lat, lon
        
        # Earth radius in meters
        R = 6371000
        
        # Convert to radians
        lat_ref = np.radians(self.gps_reference['lat'])
        lon_ref = np.radians(self.gps_reference['lon'])
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        # Calculate differences
        dlat = lat_rad - lat_ref
        dlon = lon_rad - lon_ref
        
        # Convert to local coordinates (meters)
        x_local = R * dlon * np.cos(lat_ref)
        y_local = R * dlat
        
        return x_local, y_local
        
    def imu_callback(self, msg):
        self.imu_data.append(msg)
        if len(self.imu_data) > self.window_size * 2:
            self.imu_data.popleft()
        self.run_inference()
        
    def gps_callback(self, msg):
        # Only process GPS messages with valid fix
        if msg.status.status >= 0:  # STATUS_FIX or STATUS_SBAS_FIX or STATUS_GBAS_FIX
            self.gps_data.append(msg)
            if len(self.gps_data) > self.window_size * 2:
                self.gps_data.popleft()
        
    def run_inference(self):
        # Rate limiting
        current_time = self.get_clock().now().seconds_nanoseconds()[0] + \
                      self.get_clock().now().seconds_nanoseconds()[1] * 1e-9
        if current_time - self.last_inference_time < self.min_inference_interval:
            return
        
        if len(self.imu_data) < self.window_size or len(self.gps_data) == 0:
            return
        
        try:
            # Create a window of IMU data
            imu_window = list(self.imu_data)[-self.window_size:]
            
            # Pre-compute GPS times for efficiency
            gps_times = [(gps.header.stamp.sec + gps.header.stamp.nanosec * 1e-9, gps) 
                        for gps in self.gps_data]
            
            # Find the closest GPS message for each IMU message
            processed_data = []
            for imu_msg in imu_window:
                imu_time = imu_msg.header.stamp.sec + imu_msg.header.stamp.nanosec * 1e-9
                
                # Find closest GPS with time threshold
                closest_gps = None
                min_time_diff = float('inf')
                for gps_time, gps_msg in gps_times:
                    time_diff = abs(gps_time - imu_time)
                    if time_diff < min_time_diff and time_diff <= self.max_gps_time_diff:
                        min_time_diff = time_diff
                        closest_gps = gps_msg
                
                if closest_gps:
                    # Validate GPS data
                    if not (np.isnan(closest_gps.latitude) or np.isnan(closest_gps.longitude)):
                        # Convert GPS to local coordinates
                        gps_x, gps_y = self.convert_gps_to_local(
                            closest_gps.latitude,
                            closest_gps.longitude
                        )
                        
                        processed_data.append([
                            imu_msg.orientation.x,
                            imu_msg.orientation.y,
                            imu_msg.orientation.z,
                            imu_msg.orientation.w,
                            imu_msg.angular_velocity.x,
                            imu_msg.angular_velocity.y,
                            imu_msg.angular_velocity.z,
                            imu_msg.linear_acceleration.x,
                            imu_msg.linear_acceleration.y,
                            imu_msg.linear_acceleration.z,
                            gps_x,  # Actually local x coordinate
                            gps_y,  # Actually local y coordinate
                            closest_gps.altitude,
                        ])
            
            if len(processed_data) == self.window_size:
                # Convert to numpy array directly (no DataFrame needed)
                input_array = np.array(processed_data, dtype=np.float32)
                
                # Normalize input
                input_scaled = self.X_scaler.transform(input_array)
                
                # Validate input (check for NaN/inf)
                if np.any(np.isnan(input_scaled)) or np.any(np.isinf(input_scaled)):
                    self.get_logger().warn("NaN or Inf detected in normalized input, skipping inference")
                    return
                
                # Convert to tensor and add batch dimension
                input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Run inference
                with torch.no_grad():
                    output_scaled = self.model(input_tensor)
                
                # Denormalize output
                output_scaled_np = output_scaled.squeeze().cpu().numpy()
                output = self.y_scaler.inverse_transform(output_scaled_np.reshape(1, -1))[0]
                
                # Validate output
                if np.any(np.isnan(output)) or np.any(np.isinf(output)):
                    self.get_logger().warn("NaN or Inf detected in model output, skipping publish")
                    return
                
                self.last_inference_time = current_time
                self.publish_odometry(output)
        
        except Exception as e:
            self.get_logger().error(f"Error during inference: {str(e)}", exc_info=True)
    
    def publish_odometry(self, output):
        """Publish odometry message from model output"""
        odom_msg = Odometry()
        now = self.get_clock().now().to_msg()
        odom_msg.header.stamp = now
        odom_msg.header.frame_id = "mobile_robot/odom"
        odom_msg.child_frame_id = "mobile_robot/chassis"
        
        # Parse output (assuming same order as output_labels)
        # [position_x, position_y, position_z, orientation_x, orientation_y, orientation_z, orientation_w,
        #  linear_velocity_x, linear_velocity_y, linear_velocity_z,
        #  angular_velocity_x, angular_velocity_y, angular_velocity_z]
        
        # Position
        odom_msg.pose.pose.position.x = float(output[0])
        odom_msg.pose.pose.position.y = float(output[1])
        odom_msg.pose.pose.position.z = float(output[2])
        
        # Orientation (normalize quaternion)
        q = output[3:7]
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm
        else:
            q = np.array([0, 0, 0, 1])  # Default quaternion
        
        odom_msg.pose.pose.orientation.x = float(q[0])
        odom_msg.pose.pose.orientation.y = float(q[1])
        odom_msg.pose.pose.orientation.z = float(q[2])
        odom_msg.pose.pose.orientation.w = float(q[3])
        
        # Linear velocity
        odom_msg.twist.twist.linear.x = float(output[7])
        odom_msg.twist.twist.linear.y = float(output[8])
        odom_msg.twist.twist.linear.z = float(output[9])
        
        # Angular velocity
        odom_msg.twist.twist.angular.x = float(output[10])
        odom_msg.twist.twist.angular.y = float(output[11])
        odom_msg.twist.twist.angular.z = float(output[12])
        
        # Publish
        self.fused_odom_pub.publish(odom_msg)
        
        # Broadcast transform
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = "mobile_robot/odom"
        t.child_frame_id = "mobile_robot/chassis"
        t.transform.translation.x = odom_msg.pose.pose.position.x
        t.transform.translation.y = odom_msg.pose.pose.position.y
        t.transform.translation.z = odom_msg.pose.pose.position.z
        t.transform.rotation = odom_msg.pose.pose.orientation
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
