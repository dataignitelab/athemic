FROM python:3.12-slim

RUN mkdir -p /work
WORKDIR /work
ADD ../requirements.txt ./requirements.txt
ADD dilogger.py ./dilogger.py
ADD boot.py ./boot.py
ADD ../.streamlit ./.streamlit
ADD multilanguage ./multilanguage
ADD icon ./icon
ADD post_process_algo ./post_process_algo
ADD setup_algorithm ./setup_algorithm
ADD StorageManage ./StorageManage

RUN apt-get update -y
RUN apt-get install gcc -y
RUN apt-get install g++ -y
RUN pip3 install -r requirements.txt
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "boot.py", "--server.port=8501", "--server.address=0.0.0.0"]