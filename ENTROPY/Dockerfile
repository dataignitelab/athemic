FROM python:3.12

RUN mkdir /work
WORKDIR /work
ADD StorageManage ./StorageManage
ADD boot.py ./boot.py
ADD requirements.txt ./requirements.txt

RUN apt-get update -y
#RUN apt-get install gcc -y
#RUN apt-get install g++ -y
RUN pip3 install -r requirements.txt

EXPOSE 8502
ENTRYPOINT ["streamlit", "run", "boot.py","--server.port=8502", "--server.address=0.0.0.0"]