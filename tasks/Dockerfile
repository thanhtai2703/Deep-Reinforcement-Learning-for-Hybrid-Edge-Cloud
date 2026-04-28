FROM python:3.10-slim

WORKDIR /app

COPY task_processor.py /app/task_processor.py

ENV PYTHONUNBUFFERED=1

CMD ["python", "/app/task_processor.py"]