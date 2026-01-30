FROM nvcr.io/nvidia/cuda:12.0.1-devel-ubuntu22.04

# 0) Install uv (The modern Python package manager)
# This copies the binary directly, avoiding curl/pip installation steps
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /workspace
ENV DEBIAN_FRONTEND=noninteractive

# Configure uv to install into the system python environment by default
# This allows 'uv pip install' to work like 'pip install' without a venv
ENV UV_SYSTEM_PYTHON=1

# 0.5) Update NVIDIA repo keys
# The base image is older; we need new keys to find Nsight 2026 and cuDNN 9
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends wget \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && rm -rf /var/lib/apt/lists/*

# 1) Base system dependencies + Python 3.12
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    dbus \
    fontconfig \
    gnupg \
    git \
    libasound2 \
    libfreetype6 \
    libglib2.0-0 \
    libnss3 \
    libsqlite3-0 \
    libx11-xcb1 \
    libxcb-glx0 \
    libxcb-xkb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxi6 \
    libxml2 \
    libxrandr2 \
    libxrender1 \
    libxtst6 \
    libgl1-mesa-glx \
    libxkbfile-dev \
    libmagic1 \
    libmagic-dev \
    openssh-client \
    wget \
    xcb \
    xkb-data \
    software-properties-common \
    build-essential \
    cmake \
    ninja-build \
    # Add Python 3.12 Repo
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    # Note: python3-pip and ensurepip are removed as we use uv
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# 1.5) Ensure CUDA toolkit is properly installed
# Note: The 'devel' base image already includes nvcc. 
# Only keep 'cuda-toolkit-12-0' if you explicitly need the full toolset (docs, samples, etc).
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
    cuda-toolkit-12-0 \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda-12.0
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 2) Qt6 for Nsight Compute GUI
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends qt6-base-dev \
    && rm -rf /var/lib/apt/lists/*

# 3) Python requirements using uv
COPY requirements.txt /tmp/
# uv is significantly faster and handles dependency resolution better
# We install build tools first, then the requirements
RUN uv pip install --no-cache --upgrade setuptools wheel packaging ninja \
    && uv pip install --no-cache -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# 4) cuDNN + Nsight tools
# Reverted Nsight Systems to 2025.4.1 (widely available) to fix build error
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
    libcudnn9-cuda-12 \
    nsight-compute-2025.4.1 \
    nsight-systems-2025.5.2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*