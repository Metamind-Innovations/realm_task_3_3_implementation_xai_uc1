FROM python:3.8-slim

# Set working directory
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

COPY . .

# Create component entry point
ENTRYPOINT ["python", "batch_segmentation.py"]