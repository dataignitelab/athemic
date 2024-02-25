FROM python:3.12-slim

ENV log_handler statistics-log

RUN mkdir -p /work/dataset
WORKDIR /work
ADD requirements.txt ./requirements.txt
ADD statistic_process.py /work/statistic_process.py
ADD dilogger.py /work/dilogger.py
RUN apt-get update -y
RUN apt-get install gcc -y
RUN apt-get install g++ -y
RUN pip3 install -r requirements.txt
CMD ["statistic_process.py"]
ENTRYPOINT ["python3", "-u"]