ARG BASE_IMAGE=pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
FROM ${BASE_IMAGE} as dev-base

SHELL ["/bin/bash", "-c"]

# Install required Python packages
RUN python -m pip install --upgrade pip && \
    pip install boto3 diffusers[torch] runpod && \
    pip install torchvision transformers

# Copy model directory and handler.py
COPY model /model
COPY handler.py /app/handler.py

# Set the working directory
WORKDIR /app

# Set the entry point
ENTRYPOINT ["python", "-u", "handler.py"]