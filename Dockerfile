FROM python:3.12
COPY src src/
COPY requirements.txt ./
COPY creds creds/
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install -r requirements.txt
