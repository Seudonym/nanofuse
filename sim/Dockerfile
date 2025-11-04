FROM osrf/ros:humble-desktop-full

RUN apt-get update && apt-get install -y lsb-release gnupg curl ros-humble-ros-gz bc
RUN apt-get install -y python3-pip python3-venv

RUN python3 -m venv torch_venv
RUN bash -c "source torch_venv/bin/activate && pip install torch --index-url https://download.pytorch.org/whl/cu118"
RUN bash -c "source torch_venv/bin/activate && pip install numpy joblib scikit-learn"