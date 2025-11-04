# NanoFuse

Transformer-based sensor fusion pipeline for predicting robot odometry from IMU and GPS data.

## Setup

To start the Gazebo sim:

```bash
cd sim
docker compose up --build
```

This will start publishing IMU data as JSON on a local socket, on port 5005. Look at the example code in notebooks/imu_test.ipynb for more details.

To quit, Ctrl-C and wait for it to exit.
Optionally, run for cleanup:

```bash
docker compose down
```

## Training Pipeline

### 1. Collect Data

First, start the simulation and collect sensor data:

```bash
# Terminal 1: Start simulation
cd sim
docker compose up --build

# Terminal 2: Collect data
cd notebooks
python collect_data.py
```

The data will be saved to `datasets/sensor_data_YYYYMMDD_HHMMSS.parquet`.

### 2. Preprocess Data

Preprocess the collected data with normalization and GPS coordinate conversion:

```bash
cd notebooks
python preprocess_data.py [path_to_parquet_file]
# If no file specified, uses most recent parquet file
```

This creates:
- `datasets/X_train.npy`, `datasets/X_test.npy` - Normalized input sequences
- `datasets/y_train.npy`, `datasets/y_test.npy` - Normalized output labels
- `datasets/scalers.pkl` - Normalization scalers and metadata

### 3. Train Model

Train the transformer model:

```bash
cd notebooks
python train_model.py
```

The best model will be saved to `datasets/transformer_fusion_model.pth`.

### 4. Run Inference

The inference node automatically loads the trained model and scalers. Make sure both are in the `datasets/` directory:

```bash
cd sim
docker compose up --build
```

The inference node publishes fused odometry to `/fused_odom` topic.

## Key Improvements

### Data Preprocessing
- **GPS Coordinate Conversion**: Converts GPS lat/lon to local Cartesian coordinates (meters)
- **Proper Normalization**: StandardScaler normalization for both inputs and outputs
- **Scaler Persistence**: Saves scalers for inference use

### Model Architecture
- **Better Transformer**: Multi-head attention (8 heads), positional encoding, deeper network (4 layers)
- **Improved Training**: Huber loss, learning rate scheduling, early stopping, gradient clipping
- **Better Metrics**: RMSE, MAE tracking during training

### Inference
- **Normalization Support**: Properly applies scalers during inference
- **GPS Conversion**: Converts GPS to local coordinates using saved reference
- **Updated Architecture**: Matches training model exactly

## Model Details

- **Input**: 13 features (IMU orientation, angular velocity, linear acceleration, GPS local x/y, GPS altitude)
- **Output**: 13 values (position x/y/z, orientation quaternion, linear velocity, angular velocity)
- **Window Size**: 100 timesteps
- **Architecture**: Transformer encoder with positional encoding

## Troubleshooting

**Model outputs barely move:**
- Ensure scalers are loaded correctly
- Check GPS coordinate conversion
- Verify model architecture matches training

**Training loss not decreasing:**
- Check data normalization (mean ≈ 0, std ≈ 1)
- Reduce learning rate if unstable
- Ensure sufficient training data
