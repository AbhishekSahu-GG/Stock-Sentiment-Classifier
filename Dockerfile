FROM python:3.11-slim

WORKDIR /app

RUN pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY src/ ./src/
COPY models/distilbert/ ./models/distilbert/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]