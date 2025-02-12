FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "batch_segmentation.py", \
            "--model_path", "./model_files", \
            "--path_to_test_data", "./converted_nrrds", \
            "--save_path", "./output_segmentations_test", \
            "--sensitivity", "0.9"]