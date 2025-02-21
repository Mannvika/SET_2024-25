FROM nvcr.io/nvidia/l4t-ml:r35.2.1-py3

RUN apt update && apt install --no-install-recommends -y gcc git zip curl htop libgl1-mesa-glx libglib2.0-0 libpython3-dev gnupg g++ && apt-get install --reinstall libgstreamer1.0-0

RUN mkdir -p /usr/src/ultralytics
RUN python3 -m pip install --upgrade pip wheel
RUN pip install --no-cache ultralytics --no-deps

COPY . .
ENV OMP_NUM_THREADS=1
EXPOSE 5000
CMD ["uwsgi", "--ini", "uwsgi.ini"]
