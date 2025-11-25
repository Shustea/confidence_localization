
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN useradd -m -s /bin/bash shustea
WORKDIR /workspace

# ---- system deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake gcc g++ python3-dev libffi-dev git tmux \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH

COPY --chown=shustea:shustea . .

# ---- python deps ----
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu117 && \
    pip install --no-build-isolation --no-cache-dir causal-conv1d==1.4.0 && \
    pip install --no-build-isolation --no-cache-dir pybind11>=2.11.0 mamba-ssm==2.2.2 && \
    pip install --no-cache-dir -r requirements.txt

# ---- C++ build ----
WORKDIR /workspace/data/signal_generator
RUN rm -rf build CMakeCache.txt CMakeFiles && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) && \
    make -j"$(nproc)"

USER shustea
WORKDIR /workspace
CMD ["/bin/bash"]
