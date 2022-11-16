FROM ubuntu:latest

RUN apt update
RUN apt install python3.10 -y
RUN apt install python3-pip -y
RUN apt install tesseract-ocr -y
RUN apt install vim -y

WORKDIR /usr/app/src
COPY requirements.txt ./

RUN pip install -r requirements.txt
RUN pip install fastapi[all]
RUN python3 -m spacy download en_core_web_sm

COPY project/*.py ./
RUN mkdir data
COPY project/dataset/transformers.csv ./data/
COPY project/dataset/transformer_cosine_similarity.csv ./data/

RUN python3 tester.py

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
