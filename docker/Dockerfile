FROM tensorflow/tensorflow:latest-gpu-jupyter

COPY requirements.txt .


RUN apt-get update --fix-missing && \
    apt-get install -y build-essential  && \
    apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 ffmpeg \
    git mercurial subversion && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
	
RUN pip install --upgrade pip

RUN pip install -r requirements.txt
