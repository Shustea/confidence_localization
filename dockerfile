FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Create a non-root user
RUN useradd -m -s /bin/bash shustea

# Set working directory
WORKDIR /workspace
RUN chown -R shustea:shustea /workspace

# Switch to root to install system dependencies
USER root

# Install system dependencies (added git + tmux)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libffi-dev \
    git \
    tmux \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variable for CUDA
ENV CUDA_HOME=/usr/local/cuda

# Switch to the non-root user
USER shustea

# Copy project files
COPY --chown=shustea:shustea . .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir mamba-ssm[causal-conv1d]==2.2.2

# Start with bash
CMD [ "bash" ]