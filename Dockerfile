FROM davisking/dlib:latest

RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev

RUN pip3 install flask opencv-python numpy Pillow face_recognition

WORKDIR /main
COPY . /main

ENV PORT=9000
CMD ["python3", "main.py"]
