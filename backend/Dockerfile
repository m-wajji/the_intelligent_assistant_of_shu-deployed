FROM python:3.11-slim

WORKDIR /app

ARG OPENAI_API_KEY
ARG GOOGLE_SERVICE_ACCOUNT_B64

ENV OPENAI_API_KEY=$OPENAI_API_KEY
ENV GOOGLE_SERVICE_ACCOUNT_B64=$GOOGLE_SERVICE_ACCOUNT_B64

RUN apt-get update && apt-get install -y ffmpeg

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN mkdir -p app/shu_partitioned_db/chroma_dbs

EXPOSE 8000

CMD ["uvicorn", "app.main_api:app", "--host", "0.0.0.0", "--port", "8000"]