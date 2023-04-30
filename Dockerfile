ARG BASE_IMAGE=runpod/pytorch:3.10-2.0.0-117
# ARG BASE_IMAGE=pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
FROM ${BASE_IMAGE} as dev-base

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx

# Install required Python packages
RUN python -m pip install --upgrade pip && \
    pip install opencv-python-headless boto3 diffusers[torch] runpod && \
    pip install torchvision transformers

# Copy everything else
COPY . /app

# Set the working directory
WORKDIR /app/components/CodeFormer

RUN pip install -r requirements.txt && \
    python basicsr/setup.py develop

WORKDIR /app

RUN python setup.py

# Set the entry point
ENTRYPOINT ["python", "-u", "handler.py"]