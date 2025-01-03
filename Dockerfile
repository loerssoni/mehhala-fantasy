FROM python:3.12
COPY requirements.txt ./
COPY creds creds/
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install -r requirements.txt
COPY src src/
WORKDIR src
