#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from numpy.lib.stride_tricks import as_strided
import joblib
import os
import sys

# Configuration
WINDOW_SIZE = 100  # 100 samples per window
STRIDE = 10  # Create a new window every 10 samples
DATA_DIR = "datasets"
SCALER_DIR = DATA_DIR

def load_and_prepare_data(file_path):
    """Load and prepare sensor data from parquet file"""
    print(f"Loading data from {file_path}...")
    df = pd.read_parquet(file_path)
    
    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    
    # Separate dataframes by topic
    imu_df = df[df["topic"] == "/imu"].copy()
    gps_df = df[df["topic"] == "/gps"].copy()
    odom_df = df[df["topic"] == "/odom"].copy()
    
    print(f"Found {len(imu_df)} IMU samples, {len(gps_df)} GPS samples, {len(odom_df)} Odometry samples")
    
    # Process IMU data
    imu_df = imu_df.drop(columns=["latitude", "longitude", "altitude", "position", "linear_velocity"], errors='ignore')
    
    imu_orientation = pd.DataFrame(
        imu_df["orientation"].tolist(),
        index=imu_df.index,
        columns=["imu_orientation_x", "imu_orientation_y", "imu_orientation_z", "imu_orientation_w"],
    )
    imu_angular_velocity = pd.DataFrame(
        imu_df["angular_velocity"].tolist(),
        index=imu_df.index,
        columns=["imu_angular_velocity_x", "imu_angular_velocity_y", "imu_angular_velocity_z"],
    )
    imu_linear_acceleration = pd.DataFrame(
        imu_df["linear_acceleration"].tolist(),
        index=imu_df.index,
        columns=["imu_linear_acceleration_x", "imu_linear_acceleration_y", "imu_linear_acceleration_z"],
    )
    
    imu_df = pd.concat([
        imu_df.drop(columns=["orientation", "angular_velocity", "linear_acceleration"], errors='ignore'),
        imu_orientation,
        imu_angular_velocity,
        imu_linear_acceleration,
    ], axis=1)
    
    # Process GPS data
    gps_df = gps_df.drop(columns=[
        "topic", "orientation", "angular_velocity", "linear_acceleration",
        "position", "linear_velocity"
    ], errors='ignore')
    gps_df = gps_df.rename(columns={
        "latitude": "gps_latitude",
        "longitude": "gps_longitude",
        "altitude": "gps_altitude",
    })
    
    # Process Odometry data
    odom_df = odom_df.drop(columns=["topic", "latitude", "longitude", "altitude", "linear_acceleration"], errors='ignore')
    
    odom_position = pd.DataFrame(
        odom_df["position"].tolist(),
        index=odom_df.index,
        columns=["odom_position_x", "odom_position_y", "odom_position_z"],
    )
    odom_orientation = pd.DataFrame(
        odom_df["orientation"].tolist(),
        index=odom_df.index,
        columns=["odom_orientation_x", "odom_orientation_y", "odom_orientation_z", "odom_orientation_w"],
    )
    odom_linear_velocity = pd.DataFrame(
        odom_df["linear_velocity"].tolist(),
        index=odom_df.index,
        columns=["odom_linear_velocity_x", "odom_linear_velocity_y", "odom_linear_velocity_z"],
    )
    odom_angular_velocity = pd.DataFrame(
        odom_df["angular_velocity"].tolist(),
        index=odom_df.index,
        columns=["odom_angular_velocity_x", "odom_angular_velocity_y", "odom_angular_velocity_z"],
    )
    
    odom_df = pd.concat([
        odom_df.drop(columns=["position", "orientation", "linear_velocity", "angular_velocity"], errors='ignore'),
        odom_position,
        odom_orientation,
        odom_linear_velocity,
        odom_angular_velocity,
    ], axis=1)
    
    # Sort by timestamp
    imu_df = imu_df.sort_index()
    gps_df = gps_df.sort_index()
    odom_df = odom_df.sort_index()
    
    # Merge dataframes
    merged_df = pd.merge_asof(
        imu_df, odom_df,
        left_index=True, right_index=True,
        direction="nearest",
        tolerance=pd.Timedelta("100ms"),
    )
    merged_df = pd.merge_asof(
        merged_df, gps_df,
        left_index=True, right_index=True,
        direction="nearest",
        tolerance=pd.Timedelta("500ms"),
    )
    
    # Remove rows with missing data
    merged_df = merged_df.dropna()
    print(f"After merging: {len(merged_df)} samples")
    
    return merged_df


def convert_gps_to_local(gps_df, reference_lat=None, reference_lon=None):
    """
    Convert GPS coordinates to local Cartesian coordinates (meters)
    Uses the first GPS reading as reference if not provided
    """
    if reference_lat is None:
        reference_lat = gps_df['gps_latitude'].iloc[0]
    if reference_lon is None:
        reference_lon = gps_df['gps_longitude'].iloc[0]
    
    # Earth radius in meters
    R = 6371000  # meters
    
    # Convert to radians
    lat_ref = np.radians(reference_lat)
    lon_ref = np.radians(reference_lon)
    
    lat = np.radians(gps_df['gps_latitude'].values)
    lon = np.radians(gps_df['gps_longitude'].values)
    
    # Calculate differences
    dlat = lat - lat_ref
    dlon = lon - lon_ref
    
    # Convert to local coordinates (meters)
    # This is a simple approximation valid for small distances
    x_local = R * dlon * np.cos(lat_ref)
    y_local = R * dlat
    
    gps_df = gps_df.copy()
    gps_df['gps_latitude'] = x_local
    gps_df['gps_longitude'] = y_local
    
    return gps_df, reference_lat, reference_lon


def create_windows_efficient(data, window_size, stride, input_features, output_labels):
    """Memory-efficient windowing using stride tricks"""
    input_data = data[input_features].values
    output_data = data[output_labels].values
    
    n_samples = len(input_data)
    n_input_features = len(input_features)
    
    # Calculate number of windows
    num_windows = (n_samples - window_size) // stride + 1
    
    # Create strided view for input windows
    X = as_strided(
        input_data,
        shape=(num_windows, window_size, n_input_features),
        strides=(
            stride * n_input_features * 8,
            n_input_features * 8,
            8,
        ),
        writeable=False,
    ).copy()  # Make a copy to avoid issues with views
    
    # Extract output labels at end of each window
    y_indices = np.arange(window_size - 1, window_size - 1 + num_windows * stride, stride)
    y = output_data[y_indices]
    
    return X, y


def main():
    # Get the most recent parquet file or use command line argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Find the most recent parquet file
        parquet_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.parquet')])
        if not parquet_files:
            print(f"Error: No parquet files found in {DATA_DIR}")
            sys.exit(1)
        file_path = os.path.join(DATA_DIR, parquet_files[-1])
        print(f"Using most recent file: {file_path}")
    
    # Load and prepare data
    merged_df = load_and_prepare_data(file_path)
    
    # Convert GPS to local coordinates
    print("Converting GPS coordinates to local Cartesian coordinates...")
    gps_cols = ['gps_latitude', 'gps_longitude', 'gps_altitude']
    gps_subset = merged_df[gps_cols].copy()
    gps_local, ref_lat, ref_lon = convert_gps_to_local(gps_subset)
    merged_df[gps_cols[:2]] = gps_local[gps_cols[:2]]
    
    print(f"GPS reference: lat={ref_lat:.6f}, lon={ref_lon:.6f}")
    
    # Define input and output features
    input_features = [
        "imu_orientation_x", "imu_orientation_y", "imu_orientation_z", "imu_orientation_w",
        "imu_angular_velocity_x", "imu_angular_velocity_y", "imu_angular_velocity_z",
        "imu_linear_acceleration_x", "imu_linear_acceleration_y", "imu_linear_acceleration_z",
        "gps_latitude", "gps_longitude", "gps_altitude",
    ]
    
    output_labels = [
        "odom_position_x", "odom_position_y", "odom_position_z",
        "odom_orientation_x", "odom_orientation_y", "odom_orientation_z", "odom_orientation_w",
        "odom_linear_velocity_x", "odom_linear_velocity_y", "odom_linear_velocity_z",
        "odom_angular_velocity_x", "odom_angular_velocity_y", "odom_angular_velocity_z",
    ]
    
    # Create windows
    print(f"Creating windows (window_size={WINDOW_SIZE}, stride={STRIDE})...")
    X, y = create_windows_efficient(merged_df, WINDOW_SIZE, STRIDE, input_features, output_labels)
    print(f"Created {len(X)} windows")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False  # Don't shuffle to preserve temporal order
    )
    
    # Normalize the data
    print("Normalizing data...")
    # Reshape for scaling: (samples * window_size, features)
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    
    # Fit scaler on training data
    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_test_scaled = X_scaler.transform(X_test_reshaped).reshape(X_test.shape)
    
    # Scale output data
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)
    
    # Save scalers
    os.makedirs(SCALER_DIR, exist_ok=True)
    scaler_path = os.path.join(SCALER_DIR, "scalers.pkl")
    joblib.dump({
        'X_scaler': X_scaler,
        'y_scaler': y_scaler,
        'input_features': input_features,
        'output_labels': output_labels,
        'gps_reference': {'lat': ref_lat, 'lon': ref_lon},
    }, scaler_path)
    print(f"Saved scalers to {scaler_path}")
    
    # Save normalized data
    print("Saving preprocessed data...")
    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train_scaled)
    np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test_scaled)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train_scaled)
    np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test_scaled)
    
    # Print statistics
    print("\n=== Data Statistics ===")
    print(f"Training samples: {len(X_train_scaled)}")
    print(f"Test samples: {len(X_test_scaled)}")
    print(f"Input shape: {X_train_scaled.shape}")
    print(f"Output shape: {y_train_scaled.shape}")
    print(f"\nInput feature ranges (after normalization):")
    print(f"  Mean: {X_train_scaled.mean(axis=(0,1))}")
    print(f"  Std: {X_train_scaled.std(axis=(0,1))}")
    print(f"\nOutput ranges (after normalization):")
    print(f"  Mean: {y_train_scaled.mean(axis=0)}")
    print(f"  Std: {y_train_scaled.std(axis=0)}")
    
    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
