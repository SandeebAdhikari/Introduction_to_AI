FROM python:3.8

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn pytube

COPY ./src ./

COPY ./startup.sh ./startup.sh
RUN apt-get update && apt-get install -y curl netcat-traditional
RUN chmod +x ./startup.sh 

CMD ["/bin/bash", "-c", "./startup.sh" , "unvicorn", "src.youtubeDownloader:app", "--host 0.0.0.0", "--port", "80", "--reload"]


