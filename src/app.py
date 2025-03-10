import eventlet
eventlet.monkey_patch()

from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
import cv2
import sounddevice as sd
import threading
import numpy as np
from fall_detection_system import FallDetectionSystem
import time

'''
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
discord_message(get_local_ip())'
'''

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, async_mode="eventlet", cors_allowed_origins="*")

video_frames_queue = []
audio_datas_queue = []
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024
lock = threading.Lock()
stop_event = threading.Event()  # Stop signal for threads

def capture_audio():
    """Capture audio in real-time and send to the client."""
    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        audio_datas_queue.append(indata.tolist())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=CHUNK_SIZE):
        while not stop_event.is_set():
            time.sleep(0.1)  # Prevent CPU overload

def emit_video_frames():
    """Capture video frames and send them to the client."""
    model_path = 'yolo11x-pose.pt'
    fall_system = FallDetectionSystem(model_path)
    vc = cv2.VideoCapture(0)
    if not vc.isOpened():
        print("Error: Could not open video stream.")
        return

    while not stop_event.is_set():
        rval, frame = vc.read()
        if not rval:
            break

        processed_frame = fall_system.process_frame(frame)

        with lock:
            _, encoded_image = cv2.imencode(".jpg", processed_frame)
            video_frames_queue.append(encoded_image.tobytes())

        socketio.sleep(1 / 15)  # Maintain 30 FPS

    vc.release()

def emit_data():
    """Emit video and audio data to the client."""
    while not stop_event.is_set():
        with lock:
            if video_frames_queue and audio_datas_queue:
                print('Emitting')
                socketio.emit('video_frame', {'frame': video_frames_queue[-1], 'audio_data': audio_datas_queue[-1]})
                video_frames_queue.pop(-1)
                audio_datas_queue.pop(-1)

        socketio.sleep(1 / 15)

if __name__ == '__main__':
    try:
        audio_thread = threading.Thread(target=capture_audio, daemon=True)
        video_thread = threading.Thread(target=emit_video_frames, daemon=True)
        emitter_thread = threading.Thread(target=emit_data, daemon=True)

        audio_thread.start()
        video_thread.start()
        emitter_thread.start()

        socketio.run(app, host="0.0.0.0", port=8000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)

    except KeyboardInterrupt:
        print("\nCtrl+C detected! Stopping all threads...")
        stop_event.set()
        time.sleep(1)  # Allow threads to exit gracefully
        print("Shutdown complete.")
