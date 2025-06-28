# Base image
FROM python:3.11-slim

# Cài gói hệ thống cần cho dlib, face-recognition, opencv
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Copy toàn bộ code vào image
COPY . /app

# Cài pip + thư viện Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Cổng Railway sẽ tự set qua biến môi trường
ENV PORT=9000

# Lệnh chạy Flask
CMD ["python", "main.py"]
