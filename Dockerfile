# Use official PyTorch image with CUDA 11.8 support
FROM nvidia/cuda:11.8.0-cudnn2-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip git wget ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set Python alternatives
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN python3 -m pip install --upgrade pip

# Copy the app files
WORKDIR /app
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    streamlit==1.38.0 \
    torch==2.7.0+cu118 \
    torchvision==0.22.0+cu118 \
    torchaudio==2.7.0+cu118 \
    ultralytics==8.3.186 \
    opencv-python==4.10.0.84 \
    opencv-contrib-python==4.11.0.86 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    moviepy==1.0.3

# Expose the Streamlit default port
EXPOSE 8501

# Set entrypoint
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
