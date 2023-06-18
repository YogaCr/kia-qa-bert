FROM nvidia/cuda:11.2.0-runtime-ubuntu20.04

#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

# Copy our application code
WORKDIR /app

# . Here means current directory.
COPY . .


RUN pip3 install --no-cache-dir -r requirements.txt
RUN python3 -m nltk.downloader popular

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR "/app/app"

EXPOSE 8000

# Start the app
CMD ["gunicorn", "main:app","--workers","1","-k","uvicorn.workers.UvicornWorker", "--timeout","0"]