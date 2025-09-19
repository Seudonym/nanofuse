# NanoFuse

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
