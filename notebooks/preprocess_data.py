#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the collected sensor data
file_path = "/home/wahid/dev/nanofuse/datasets/sensor_data_20251023_152921.parquet"
df = pd.read_parquet(file_path)

# Display the first few rows of the dataframe
print("First 5 rows of the dataset:")
df.head()


# In[2]:


df.topic.unique()


# In[3]:


df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")
df.head()


# In[4]:


# Separate dataframes by topic
imu_df = df[df["topic"] == "/imu"].copy()
gps_df = df[df["topic"] == "/gps"].copy()
odom_df = df[df["topic"] == "/odom"].copy()


# In[5]:


imu_df = imu_df.drop(
    columns=["latitude", "longitude", "altitude", "position", "linear_velocity"]
)
imu_df.head()


# In[6]:


imu_orientation = pd.DataFrame(
    imu_df["orientation"].tolist(),
    index=imu_df.index,
    columns=[
        "imu_orientation_x",
        "imu_orientation_y",
        "imu_orientation_z",
        "imu_orientation_w",
    ],
)
imu_angular_velocity = pd.DataFrame(
    imu_df["angular_velocity"].tolist(),
    index=imu_df.index,
    columns=[
        "imu_angular_velocity_x",
        "imu_angular_velocity_y",
        "imu_angular_velocity_z",
    ],
)
imu_linear_acceleration = pd.DataFrame(
    imu_df["linear_acceleration"].tolist(),
    index=imu_df.index,
    columns=[
        "imu_linear_acceleration_x",
        "imu_linear_acceleration_y",
        "imu_linear_acceleration_z",
    ],
)


# In[7]:


imu_df = pd.concat(
    [
        imu_df.drop(columns=["orientation", "angular_velocity", "linear_acceleration"]),
        imu_orientation,
        imu_angular_velocity,
        imu_linear_acceleration,
    ],
    axis=1,
)

imu_df.head()


# In[8]:


gps_df = gps_df.drop(
    columns=[
        "topic",
        "orientation",
        "angular_velocity",
        "linear_acceleration",
        "position",
        "linear_velocity",
        "angular_velocity",
    ]
)
gps_df = gps_df.rename(
    columns={
        "latitude": "gps_latitude",
        "longitude": "gps_longitude",
        "altitude": "gps_altitude",
    }
)
gps_df.head()


# In[9]:


odom_df = odom_df.drop(
    columns=["topic", "latitude", "longitude", "altitude", "linear_acceleration"]
)
odom_df.head()


# In[10]:


odom_position = pd.DataFrame(
    odom_df["position"].tolist(),
    index=odom_df.index,
    columns=["odom_position_x", "odom_position_y", "odom_position_z"],
)
odom_orientation = pd.DataFrame(
    odom_df["orientation"].tolist(),
    index=odom_df.index,
    columns=[
        "odom_orientation_x",
        "odom_orientation_y",
        "odom_orientation_z",
        "odom_orientation_w",
    ],
)
odom_linear_velocity = pd.DataFrame(
    odom_df["linear_velocity"].tolist(),
    index=odom_df.index,
    columns=[
        "odom_linear_velocity_x",
        "odom_linear_velocity_y",
        "odom_linear_velocity_z",
    ],
)
odom_angular_velocity = pd.DataFrame(
    odom_df["angular_velocity"].tolist(),
    index=odom_df.index,
    columns=[
        "odom_angular_velocity_x",
        "odom_angular_velocity_y",
        "odom_angular_velocity_z",
    ],
)


# In[11]:


odom_df = pd.concat(
    [
        odom_df.drop(
            columns=["position", "orientation", "linear_velocity", "angular_velocity"]
        ),
        odom_position,
        odom_orientation,
        odom_linear_velocity,
        odom_angular_velocity,
    ],
    axis=1,
)
odom_df.head()


# In[12]:


imu_df = imu_df.sort_index()
gps_df = gps_df.sort_index()
odom_df = odom_df.sort_index()


# In[13]:


merged_df = pd.merge_asof(
    imu_df,
    odom_df,
    left_index=True,
    right_index=True,
    direction="nearest",
    tolerance=pd.Timedelta("100ms"),
)
merged_df = pd.merge_asof(
    merged_df,
    gps_df,
    left_index=True,
    right_index=True,
    direction="nearest",
    tolerance=pd.Timedelta("500ms"),
)


# In[14]:


merged_df = merged_df.dropna()


# In[15]:


merged_df.info()


# In[16]:


merged_df.head()


# In[ ]:


# In[ ]:


# In[18]:


import numpy as np


# In[19]:


input_features = [
    "imu_orientation_x",
    "imu_orientation_y",
    "imu_orientation_z",
    "imu_orientation_w",
    "imu_angular_velocity_x",
    "imu_angular_velocity_y",
    "imu_angular_velocity_z",
    "imu_linear_acceleration_x",
    "imu_linear_acceleration_y",
    "imu_linear_acceleration_z",
    "gps_latitude",
    "gps_longitude",
    "gps_altitude",
]

output_labels = [
    "odom_position_x",
    "odom_position_y",
    "odom_position_z",
    "odom_orientation_x",
    "odom_orientation_y",
    "odom_orientation_z",
    "odom_orientation_w",
    "odom_linear_velocity_x",
    "odom_linear_velocity_y",
    "odom_linear_velocity_z",
    "odom_angular_velocity_x",
    "odom_angular_velocity_y",
    "odom_angular_velocity_z",
]


# In[20]:


def create_windows(data, window_size, stride, input_features, output_labels):
    X, y = [], []
    num_samples = len(data)
    for i in range(0, num_samples - window_size, stride):
        # Input window is the sequence of sensor data
        X.append(data[input_features].iloc[i : i + window_size].values)
        # Output is the ground truth at the end of the window
        y.append(data[output_labels].iloc[i + window_size - 1].values)
    return np.array(X), np.array(y)


from numpy.lib.stride_tricks import as_strided


# Define window parameters
window_size = 100  # 100 samples per window (0.5 seconds at 200Hz)
stride = 10  # Create a new window every 10 samples


def create_windows_efficient(data, window_size, stride, input_features, output_labels):
    """Memory-efficient windowing using stride tricks - creates views instead of copies"""

    # Extract input and output arrays
    input_data = data[input_features].values
    output_data = data[output_labels].values

    n_samples = len(input_data)
    n_input_features = len(input_features)
    n_output_features = len(output_labels)

    # Calculate number of windows
    num_windows = (n_samples - window_size) // stride + 1

    # Create strided view for input windows (no data duplication!)
    X = as_strided(
        input_data,
        shape=(num_windows, window_size, n_input_features),
        strides=(
            stride * n_input_features * 8,
            n_input_features * 8,
            8,
        ),  # 8 bytes for float64
        writeable=False,
    )

    # Extract output labels at end of each window
    y_indices = np.arange(
        window_size - 1, window_size - 1 + num_windows * stride, stride
    )
    y = output_data[y_indices]

    return X, y


# Use the efficient version
X, y = create_windows_efficient(
    merged_df, window_size, stride, input_features, output_labels
)
# In[21]:

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
np.save("/home/wahid/dev/nanofuse/datasets/X_train.npy", X_train)
np.save("/home/wahid/dev/nanofuse/datasets/X_test.npy", X_test)
np.save("/home/wahid/dev/nanofuse/datasets/y_train.npy", y_train)
np.save("/home/wahid/dev/nanofuse/datasets/y_test.npy", y_test)
