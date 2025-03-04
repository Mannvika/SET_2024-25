from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
import cv2
import multiprocessing
import numpy as np
import socket
import requests
import time
from AudioClassifier import AudioClassifier  # Your scream detection class
from fall_detection_system import FallDetectionSystem  # Your fall detection class

# Discord Webhook for Notifications
WEBHOOK = "https://discord.com/api/webhooks/1329639907442036769/5ShE26g-ZleAN1lY7L5lPGv-HyqZx7TukNTF2rrAwuQeWNUku4dNMrsWZBnHKnJYZOlN"

def get_local_ip():
    """Get local network IP address"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        s.connect(("8.8.8.8", 80))  # Connect to Google's DNS
        ip = s.getsockname()[0]
    except Exception:
        ip = "Unable to determine local IP"
    finally:
        s.close()
    return ip

def discord_message(ip):
    """Send local IP address to Discord webhook"""
    message = {"embeds": [{"title": "IP Address", "color": 65280, "description": f"{ip}:8000"}]}
    x = requests.post(WEBHOOK, json=message)
    print("Discord Notification:", "Success" if x.status_code == 204 else "Failed")

# Initialize Flask & SocketIO
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

def emit_video_frames(frame_queue):
    """Capture video frames and send them to the client via a queue"""
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
        
        processed_frame = fall_system.process_frame(frame)  # Process frame
        _, encoded_image = cv2.imencode(".jpg", processed_frame)
        frame_queue.put(encoded_image.tobytes())  # Send frame to main process

    vc.release()

def audio_process(audio_queue):
    """Runs scream detection in a separate process"""
    device_id = 0  # Adjust as needed
    scream_detector = AudioClassifier(device_id)
    scream_detector.start_listening(audio_queue)

if __name__ == '__main__':
    # Start Processes
    frame_queue = multiprocessing.Queue()
    audio_queue = multiprocessing.Queue()

    video_process = multiprocessing.Process(target=emit_video_frames, args=(frame_queue,))
    audio_process = multiprocessing.Process(target=audio_process, args=(audio_queue,))

    video_process.start()
    audio_process.start()

    print("Local Network IP Address:", get_local_ip())
    discord_message(get_local_ip())

    @socketio.on('connect')
    def handle_connect():
        print("Client Connected")

    def send_video_frames():
        """Continuously send video frames to client"""
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                socketio.emit('video_frame', {'frame': frame})

    def send_audio_alerts():
        """Send scream detection alerts to client"""
        while True:
            if not audio_queue.empty():
                alert = audio_queue.get()
                socketio.emit('audio_alert', {'message': alert})

    # Start background tasks for emitting data
    socketio.start_background_task(send_video_frames)
    socketio.start_background_task(send_audio_alerts)

    # Run Flask app
    socketio.run(app, host="0.0.0.0", port=8000, debug=False, use_reloader=False)
