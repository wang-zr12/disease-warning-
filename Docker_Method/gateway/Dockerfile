FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install fastapi uvicorn requests
CMD ["python", "app.py"]
