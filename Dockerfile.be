FROM python:3.9-slim as builder

#set up environment
# RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
# RUN apt-get install unzip
# RUN apt-get -y install python3
# RUN apt-get -y install python3-pip

# Copy our application code
WORKDIR /app

# . Here means current directory.
COPY . .


RUN pip3 install --no-cache-dir -r requirements.txt
RUN python3 -m nltk.downloader popular

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR "/app/app"

EXPOSE 5421

# Start the app
CMD ["gunicorn","--bind","0.0.0.0:5421", "main:app", "--timeout","0"]