FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt /app/
RUN sudo apt update
RUN apt install libatlas-base-dev libportaudio2 libasound2-dev
RUN apt update && apt install -y portaudio19-dev python3-pyaudio
RUN pip install pyaudio
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install edge_impulse_linux
# Copy Python files
COPY src/*.py /app/

CMD ["python3", "app.py"]
