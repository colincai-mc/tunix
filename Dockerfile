# Base image with Python 3.12
FROM python:3.12-slim

# Set environment variables to non-interactive to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies, including Python 3 and pip
RUN apt-get update && \
    apt-get install -y build-essential git python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Create a virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

RUN pip install git+https://github.com/ayaka14732/jax-smi.git
# If you encounter a checkpoint issue, try using following old version of pathways-utils.
# RUN pip install git+https://github.com/AI-Hypercomputer/pathways-utils.git@b72729bb152b7b3426299405950b3af300d765a9#egg=pathwaysutils
RUN pip install gcsfs
RUN pip install wandb
# gymnasium is needed by examples/frozenlake/env.py (not pulled in by base deps).
RUN pip install gymnasium

ENV VLLM_TARGET_DEVICE=tpu

# Build vllm + tpu-inference from source using the upstream-supported flow:
#   1) clone, 2) install vllm/requirements/tpu.txt (sets up torch_xla / jax), 3) editable install.
# `pip install vllm @ git+...` does not replicate this and the cmake step fails.
# Done BEFORE COPY . . so code edits don't bust the cmake cache.
ARG VLLM_COMMIT=5e584ce9ecb3cce63f1caab86177aef5c831690f
ARG TPU_INFERENCE_COMMIT=1d3cd6ed68f5576a18bdff2dee6e4e28f3c251bb

WORKDIR /usr/src
RUN git clone https://github.com/vllm-project/vllm.git && \
    cd vllm && git checkout ${VLLM_COMMIT} && \
    pip uninstall torch torch-xla -y || true && \
    pip install -r requirements/tpu.txt && \
    python -m pip install -e . --no-build-isolation

WORKDIR /usr/src
RUN git clone https://github.com/vllm-project/tpu-inference.git && \
    cd tpu-inference && git checkout ${TPU_INFERENCE_COMMIT} && \
    pip install -r requirements.txt && pip install -e .

# Tpu-inference pins qwix to 0.1.2 causing lora issues.
RUN pip install --no-deps "qwix>=0.1.6"

# Transformers 5.5.3 is required for `gemma4` model type (tpu_inference pin).
RUN pip install --upgrade "transformers==5.5.3"

# Project copy + editable install last, so code edits only invalidate this layer.
WORKDIR /app
COPY . .
RUN pip install -e .

# Set the default command to bash
CMD ["bash"]
