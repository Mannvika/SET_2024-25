FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy Python files
COPY src/*.py /app/

CMD ["python3", "app.py"]
