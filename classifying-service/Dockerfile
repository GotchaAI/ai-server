FROM python:3.9-slim
WORKDIR /app

COPY requirements.txt .
COPY ./models ./models

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]