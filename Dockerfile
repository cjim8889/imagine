ARG BASE_IMAGE=runpod/pytorch:3.10-2.0.0-117
FROM ${BASE_IMAGE} as dev-base

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y build-essential cmake
# Install required Python packages
RUN python -m pip install --upgrade pip && \
    pip install boto3 diffusers[torch] runpod && \
    pip install torchvision transformers

RUN pip install dlib

# Copy model directory and handler.py
COPY . /app

# Set the working directory
WORKDIR /app/components/CodeFormer


RUN pip install -r requirements.txt && \
    python basicsr/setup.py develop

WORKDIR /app

# Set the entry point
ENTRYPOINT ["python", "-u", "handler.py"]