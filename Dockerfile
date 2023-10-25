# app/Dockerfile

FROM python:3.9-slim

# WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# COPY . /app

WORKDIR /app 

RUN git clone https://github.com/sudharsan-007/playground . 

RUN pip3 install -r requirements.txt

EXPOSE 8002

# HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "01_🧩_Playgorund.py", "--server.port=8002", "--server.address=0.0.0.0"]