FROM python:3.12-slim

RUN apt-get update
RUN mkdir -p /work/dataset
WORKDIR /work
ADD requirements.txt ./requirements.txt
ADD dilogger.py ./dilogger.py
ADD RandomForestRegression.py /work/RandomForestRegression.py

RUN apt-get update -y
RUN apt-get install gcc -y
RUN apt-get install g++ -y
RUN pip install -r requirements.txt

CMD ["RandomForestRegression.py"]
ENTRYPOINT ["python3", "-u"]