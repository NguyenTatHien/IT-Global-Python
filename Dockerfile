FROM python:3.11-slim

# Cài các gói hệ thống cho dlib, face_recognition, opencv
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
WORKDIR /main

# Copy code vào container
COPY . /main

# Cài thư viện Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Railway truyền PORT qua biến môi trường
ENV PORT=9000

CMD ["python", "main.py"]
