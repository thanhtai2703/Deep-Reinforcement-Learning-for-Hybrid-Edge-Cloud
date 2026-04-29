FROM python:3.10-slim

WORKDIR /app

COPY task-processor.py /app/task_processor.py

ENV PYTHONUNBUFFERED=1

CMD ["python", "/app/task_processor.py"]