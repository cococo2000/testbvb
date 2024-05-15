FROM ubuntu:22.04

RUN sed -i 's/http:\/\/archive.ubuntu.com/http:\/\/mirrors.tuna.tsinghua.edu.cn\/ubuntu/g' /etc/apt/sources.list
RUN apt-get clean
RUN apt-get update
RUN apt-get install -y python3-numpy python3-scipy python3-pip build-essential git
RUN pip3 install -U pip
RUN python3 --version | grep 'Python 3.10'

WORKDIR /home/app
COPY requirements.txt run_algorithm.py ./
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3", "-u", "run_algorithm.py"]
