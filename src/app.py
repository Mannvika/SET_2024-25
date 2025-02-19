from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO
import cv2
import sounddevice as sd
import numpy as np
import multiprocessing
import threading
from fall_detection_system import FallDetectionSystem
import socket
import requests

webhook = "https://discord.com/api/webhooks/1329639907442036769/5ShE26g-ZleAN1lY7L5lPGv-HyqZx7TukNTF2rrAwuQeWNUku4dNMrsWZBnHKnJYZOlN"

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        s.connect(("8.8.8.8", 80))  # Connects to Google DNS (doesn't actually send data)
        ip = s.getsockname()[0]
    except Exception:
        ip = "Unable to determine local IP"
    finally:
        s.close()
    return ip

def discord_message(ip):
    message = {
        "embeds": [{
        "title": f"IP Address",
        "color": 65280,
        "description": f"{ip}:8000"
        }]
    }

    x = requests.post(webhook, json=message)
    if x.status_code == 204:
        print("success")
    else:
        print("failed")

print("Local Network IP Address:", get_local_ip())
discord_message(get_local_ip())


app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")



# Video and audio parameters
SAMPLE_RATE = 44100  # Audio sample rate in Hz
CHUNK_SIZE = 1024  # Audio chunk size
lock = threading.Lock()

def capture_audio():
    """Capture audio in real-time and send to the client."""
    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        # Send audio data to React client
        socketio.emit('audio_data', indata.tolist())

    # Start audio stream
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=CHUNK_SIZE):
        threading.Event().wait()  # Keep thread running

@app.route('/stream', methods=['GET'])
def stream():
    """Stream video frames to the client."""
    return Response(generate_video(), mimetype="multipart/x-mixed-replace; boundary=frame")

def generate_video():
    """Generate video frames for streaming."""
    model_path = 'yolo11x-pose.pt'
    fall_system = FallDetectionSystem(model_path)
    vc = cv2.VideoCapture(0)
    if not vc.isOpened():
        return

    while True:
        rval, frame = vc.read()
        # If frame capture was unsuccessful, break the loop
        if not rval:
            break
        # Run pose estimation on the captured frame
        processed_frame = fall_system.process_frame(frame)
        with lock:
            _, encoded_image = cv2.imencode(".jpg", processed_frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')
    vc.release()

@socketio.on('audio_data_from_client')
def handle_audio_data_from_client(data):
    """Receive audio data from the client."""
    print("Received audio data from client:", data)
    # Process or save the incoming audio data as needed

if __name__ == '__main__':
    # Start audio capture in a separate thread
    audio_process = multiprocessing.Process(target=capture_audio, daemon=True)
    audio_process.start()
    # Run the Flask-SocketIO app
    print("Local Network IP Address:", get_local_ip())
    socketio.run(app, host="0.0.0.0", port=8000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
