from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO
import cv2
import threading
from fall_detection_system import FallDetectionSystem
from dynamo import save_to_dynamo
import random
import sounddevice as sd
import numpy as np
import multiprocessing
import socket
import requests
import time
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

def log_inference_time(inference_time):
    """Send inference log to frontend via SocketIO"""
    print(f"Sending log: Inference time: {inference_time}ms")  # Print statement before sending log
    socketio.emit("log", {"message": f"Inference time: {inference_time}ms"})
    print(f"Log sent: Inference time: {inference_time}ms")  # Confirm the log is sent

def log_message(message):
    """Helper function to emit logs to frontend"""
    print(message)  # Log to console
    socketio.emit('log', {'message': message}, broadcast=True)  # Send log to React

# @app.route('/stream', methods=['GET'])
# def stream():
#     """Stream video frames to the client."""
#     return Response(generate_video(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/save_data', methods=['POST'])
def save_data():
    """Handles data saving process"""
    global saving_data
    command = request.get_json()

    if not command or 'action' not in command:
        log_message("Invalid request received")
        return jsonify({"message": "Invalid request"}), 400

    action = command['action']

    if action == 'start':
        if not saving_data:
            saving_data = True
            log_message("Started saving data...")
            data = {"runID": "A1"}  # Hardcoded runID
            save_to_dynamo(data)  # Send data to DynamoDB
            return jsonify({"message": "Data saving started"}), 200
        else:
            log_message("Data saving is already active.")
            return jsonify({"message": "Data saving is already active"}), 200

    elif action == 'stop':
        if saving_data:
            saving_data = False
            log_message("Stopped saving data.")
            return jsonify({"message": "Data saving stopped"}), 200
        else:
            log_message("No active data saving to stop.")
            return jsonify({"message": "No active data saving to stop"}), 200

    else:
        log_message("Invalid action received.")
        return jsonify({"message": "Invalid action"}), 400
#
# def generate_video():
#     """Generate video frames for streaming."""
#     model_path = 'yolo11x-pose.pt'
#     fall_system = FallDetectionSystem(model_path)
#     vc = cv2.VideoCapture(0)
#
#     if not vc.isOpened():
#         log_message("Error: Unable to open camera.")
#         return
#
#     while True:
#         rval, frame = vc.read()
#         if not rval:
#             log_message("Error: Frame capture failed.")
#             break
#
#         # Simulate inference time
#         # Sample code for displaying live messages to react frontend
#         inference_time = random.uniform(590, 620)  # Example log value
#         log_inference_time(inference_time)  # Emit log to frontend
#
#
#         # Run pose estimation on the captured frame
#         processed_frame = fall_system.process_frame(frame)
#
#         with lock:
#             _, encoded_image = cv2.imencode(".jpg", processed_frame)
#             yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')
#
#     vc.release()





def capture_audio():
    """Capture audio in real-time and send to the client."""

    def audio_callback(indata, frames, time, status):
        print(status)
        if status:
            print(status)
        # Send audio data to React client
        socketio.emit('audio_data', indata.tolist())

    # Start audio stream
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=CHUNK_SIZE):
        threading.Event().wait()  # Keep thread running


def emit_video_frames():
    """Capture video frames and send them to the client via SocketIO."""
    model_path = 'yolo11x-pose.pt'
    fall_system = FallDetectionSystem(model_path)
    vc = cv2.VideoCapture(0)

    if not vc.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        rval, frame = vc.read()
        if not rval:
            break

        # Run pose estimation on the captured frame
        processed_frame = fall_system.process_frame(frame)

        with lock:
            _, encoded_image = cv2.imencode(".jpg", processed_frame)
            socketio.emit('video_frame', {'frame': encoded_image.tobytes()})

        time.sleep(1 / 30)  # Adjust to match the desired FPS (30 FPS)

    vc.release()


def generate_audio_data():
    """Simulates audio data streaming from the Flask app."""
    while True:
        # Simulated audio data (replace this with actual audio stream data)
        audio_chunk = np.random.randint(-32768, 32767, 1024, dtype=np.int16).tobytes()
        socketio.emit('audio_data_to_client', {'audio': audio_chunk})
        time.sleep(0.1)  # Simulatinsg real-time streaming


if __name__ == '__main__':
    # Start audio capture in a separate thread
    audio_thread = threading.Thread(target=capture_audio, daemon=True)
    audio_thread.start()

    video_thread = threading.Thread(target=emit_video_frames, daemon=True)
    video_thread.start()
    # Run the Flask-SocketIO app
    print("Local Network IP Address:", get_local_ip())
    socketio.run(app, host="0.0.0.0", port=8000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
# if __name__ == '__main__':
#     socketio.run(app, host="0.0.0.0", port=8000, debug=True, allow_unsafe_werkzeug=True)
