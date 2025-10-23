import socket
import json
import pandas as pd
import time
from datetime import datetime
import signal
import os

HOST, PORT = "0.0.0.0", 5005
SAVE_INTERVAL = 60  # seconds
OUTPUT_DIR = "datasets"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"sensor_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet")

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))

print(f"Listening for sensor data on {HOST}:{PORT}...")
print(f"Data will be saved to {OUTPUT_FILE}")

data_list = []
last_save_time = time.time()

def save_data(signum=None, frame=None):
    if not data_list:
        print("No new data to save.")
        if signum is not None:
            exit(0)
        return

    print(f"Saving {len(data_list)} data points to {OUTPUT_FILE}...")
    df = pd.DataFrame(data_list)
    if os.path.exists(OUTPUT_FILE):
        # If file exists, read it and append new data
        existing_df = pd.read_parquet(OUTPUT_FILE)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df.to_parquet(OUTPUT_FILE, engine='pyarrow', compression='snappy')
    else:
        df.to_parquet(OUTPUT_FILE, engine='pyarrow', compression='snappy')
    data_list.clear()
    print("Save complete.")
    if signum is not None:
        print("Exiting.")
        exit(0)

# Register signal handler for graceful shutdown
signal.signal(signal.SIGINT, save_data)
signal.signal(signal.SIGTERM, save_data)

try:
    while True:
        data, _ = sock.recvfrom(65535)
        msg = json.loads(data.decode())
        
        # Add a timestamp
        msg['timestamp'] = datetime.now().isoformat()
        
        data_list.append(msg)
        
        # Check if it's time to save
        if time.time() - last_save_time > SAVE_INTERVAL:
            save_data()
            last_save_time = time.time()

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    save_data()
    sock.close()
    print("Socket closed.")
