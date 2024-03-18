FROM python:3.8

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn pytube

EXPOSE 80

COPY ./startup.sh ./startup.sh

RUN apt-get update && apt-get install -y curl netcat-traditional
RUN chmod +x ./startup.sh

COPY ./src ./src

CMD ["/bin/bash", "-c", "./startup.sh & uvicorn src.videoDownloader:app --host 0.0.0.0 --port 80"]

