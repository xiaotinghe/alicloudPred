FROM python:3.6
RUN mkdir -p /app
COPY ./inference /app/inference
RUN python -m pip install -r /app/inference/requirements.txt