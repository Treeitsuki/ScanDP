# ベースイメージ
FROM osrf/ros:noetic-desktop-focal

# 環境変数
ENV DEBIAN_FRONTEND=noninteractive

# 基本的なパッケージ
RUN apt-get update && apt-get install -y \
    ros-noetic-desktop-full=1.5.0-1* \
    mesa-utils \
    x11-apps \
    libgl1-mesa-glx \
    libglu1-mesa \
    libx11-dev \
    libusb-1.0-0-dev \
    usbutils \
    sudo \
    vim \
    iputils-ping \
    net-tools \
    sshpass \
    udev \
    wget \
    curl \
    git \
    python3-pip \
    python3-catkin-tools \
    python3-rosinstall-generator \
    && rm -rf /var/lib/apt/lists/*

# OpenGLとNVIDIA関連のパッケージをインストール
RUN apt-get update && apt-get install -y \
    nvidia-cuda-toolkit \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    liburdfdom-tools \
    && rm -rf /var/lib/apt/lists/*

# ROSパッケージをインストール
RUN apt-get update && apt-get install -y \
    ros-noetic-ros-control \
    ros-noetic-rtabmap \
    ros-noetic-rtabmap-ros \
    ros-noetic-pcl-ros \
    ros-noetic-realsense2-camera \
    ros-noetic-moveit* \
    ros-noetic-ros-control \
    ros-noetic-ros-controllers \
    ros-noetic-joint-state-controller \
    ros-noetic-effort-controllers \
    ros-noetic-position-controllers \
    ros-noetic-joint-trajectory-controller \
    ros-noetic-tf2-sensor-msgs \
    && rm -rf /var/lib/apt/lists/*

# Install build dependencies and librealsense v2.50.0
RUN apt-get update && apt-get install -y \
    git cmake build-essential libssl-dev libusb-1.0-0-dev pkg-config \
    libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --depth 1 --branch v2.50.0 https://github.com/IntelRealSense/librealsense.git /tmp/librealsense \
    && cd /tmp/librealsense \
    && mkdir -p build && cd build \
    && cmake .. \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && rm -rf /tmp/librealsense

# Install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin:$PATH"

# [追加] Node.js と Codex CLI のインストール
# RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash && \
#     \. "$HOME/.nvm/nvm.sh" && \
#     nvm install 20 && \
#     nvm use 20 && \
#     # Codex CLIをグローバルインストール
#     npm install -g @openai/codex && \
#     rm -rf /var/lib/apt/lists/*

# ユーザー作成用の引数
ARG UID=1000
ARG GID=1000
ARG USERNAME=user
ARG GROUPNAME=user
ARG PASSWORD=user

# ユーザーとグループを作成
# Ensure a system 'video' group exists and add the user to it so they can access /dev/video* when the
# container runtime exposes the host devices. Also provide passwordless sudo for convenience.
RUN groupadd -g $GID $GROUPNAME && \
    # create a 'video' group if it doesn't already exist (use system/groupadd -r) and ignore failures
    groupadd -r video || true && \
    useradd -m -s /bin/bash -u $UID -g $GID -G sudo,video $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# 作業ディレクトリを作成
RUN mkdir -p /home/$USERNAME/workspace
WORKDIR /home/$USERNAME/workspace

# requirements.txtをコピーしてPythonパッケージをインストール
COPY requirements.txt /tmp/requirements.txt
RUN if [ -s /tmp/requirements.txt ]; then pip install -r /tmp/requirements.txt; fi

# ROS環境設定をbashrcに追加
RUN echo "source /opt/ros/noetic/setup.bash" >> /home/$USERNAME/.bashrc && \
    echo "source ~/workspace/devel/setup.bash" >> /home/$USERNAME/.bashrc && \
    echo "alias i611='ssh i611usr@192.168.0.23'" >> /home/$USERNAME/.bashrc

# rviz
RUN echo "export __NV_PRIME_RENDER_OFFLOAD=1" >> /home/$USERNAME/.bashrc && \
    echo "export __GLX_VENDOR_LIBRARY_NAME=nvidia" >> /home/$USERNAME/.bashrc && \
    echo "export DISABLE_ROS1_EOL_WARNINGS=1" >> /home/$USERNAME/.bashrc

# ワークスペースの所有権を変更
RUN chown -R $USERNAME:$GROUPNAME /home/$USERNAME/workspace

# Create the Mesa shader cache directory and ensure the non-root user owns it so the
# runtime can write shader cache files (fixes "Failed to create /home/user/.cache/mesa_shader_cache").
RUN mkdir -p /home/$USERNAME/.cache/mesa_shader_cache && \
    chown -R $USERNAME:$GROUPNAME /home/$USERNAME/.cache

# 通常ユーザーに切り替え
USER $USERNAME

# デフォルトで bash を起動
CMD ["/bin/bash"]