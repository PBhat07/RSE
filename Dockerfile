# Use the official NVIDIA CUDA image as the base
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set a working directory inside the container
WORKDIR /opt/GaussianAvatars



# Set up the environment and install dependencies.
# The `RUN` command combines multiple steps into a single layer for efficiency.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        curl \
        build-essential \
        git \
        python3-distutils \
        python3-dev \
        libgl1-mesa-glx \
        libglx-mesa0 \
        libglib2.0-0 \
        libxext6 \
        libx11-6 \
        libsm6 \
        libxrender1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Set the PATH to include Conda
ENV PATH="/opt/conda/bin:$PATH"

# Accept the Conda Terms of Service for non-interactive builds
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Set architecture flags BEFORE installing any packages with CUDA extensions
ENV TORCH_CUDA_ARCH_LIST="8.9"

# Copy requirements.txt separately to allow pip install to be cached
COPY requirements.txt .
COPY submodules ./submodules

RUN conda create --name gaussian-avatars -y python=3.10 && \
    /opt/conda/envs/gaussian-avatars/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    conda install -n gaussian-avatars -c conda-forge libstdcxx-ng -y && \
    /opt/conda/envs/gaussian-avatars/bin/pip install setuptools==67.8.0 ninja && \
    /opt/conda/envs/gaussian-avatars/bin/pip install -r requirements.txt && \
    git clone --depth=1 https://github.com/leo-frank/diff-gaussian-rasterization-depth.git /opt/diff-gaussian-rasterization-depth && \
    cd /opt/diff-gaussian-rasterization-depth && \
    /opt/conda/envs/gaussian-avatars/bin/pip install . && \
    rm -rf /opt/diff-gaussian-rasterization-depth/.git


# This layer will be rebuilt on every code change, but the previous layers will be reused from the cache
COPY . .


# Set the environment variables for the container
ENV PATH="/opt/conda/envs/gaussian-avatars/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/conda/envs/gaussian-avatars/lib:/opt/conda/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH"

# This is the fix: The command sources the Conda initialization script and then activates the environment.
CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate gaussian-avatars && /bin/bash"]
