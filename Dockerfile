# Ubuntu Linux as the base image
FROM ubuntu:22.04

# Set UTF-8 encoding
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Install Python
RUN apt-get -y update && \
    apt-get -y upgrade
# The following line ensures that the subsequent install doesn't expect user input
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y install python3-pip python3-dev

# Install spaCy
RUN pip3 install --upgrade pip
RUN pip3 install spacy
RUN python3 -m spacy download en_core_web_lg
RUN python3 -m spacy download en_core_web_sm

# Install datasets transformers evaluate
RUN pip install datasets transformers evaluate

# Install nltk
RUN pip install nltk

# Install gdown
RUN pip install gdown

# Install torch
RUN pip3 install torchvision

#Install sentencepiece
RUN pip install sentencepiece

#Download punkt
#RUN nltk.download("punkt")

# Install protobuf
RUN pip install --upgrade "protobuf<=3.20.1"
# RUN export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

#Install sentence_transformers and huggingface
RUN pip install sentence_transformers
RUN pip install huggingface_hub
RUN pip3 install boolean-question

# Add the files into container, under QA folder, modify this based on your need
RUN mkdir /QA
ADD ask /QA
ADD answer /QA

# Change the permissions of programs
CMD ["chmod 777 /QA/*"]

# Set working dir as /QA
WORKDIR /QA
ENTRYPOINT ["/bin/bash", "-c"]
