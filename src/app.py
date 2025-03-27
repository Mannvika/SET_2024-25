import eventlet
eventlet.monkey_patch()

from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
import cv2
import sounddevice as sd
import numpy as np
from fall_detection_system import FallDetectionSystem
import time
from collections import deque

# From Communications
from edge_impulse_linux.audio import AudioImpulseRunner
from AudioClassifier import AudioClassifier
import queue

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, async_mode="eventlet", cors_allowed_origins="*")

video_frames_queue = deque()
audio_datas_queue = deque()
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024
lock = eventlet.semaphore.Semaphore()
stop_event = eventlet.event.Event()

compressFrame = False
device_id = 0  # Change if needed

MODEL_PATH = "/home/ufset/Desktop/SET_2024-25/src/audio_model.eim"

def capture_audio():
    """Capture audio in real-time and send to the client."""
    audio_classifier = AudioClassifier(device_id)  # Change as needed

    def audio_callback(indata, frames, time, status):
        if status:
            print(status)

        with lock:
            audio_datas_queue.append(indata.tolist())
            audio_classifier.audio_queue.put(indata.tolist())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=CHUNK_SIZE):
        runner = AudioImpulseRunner(MODEL_PATH)
        try:
            model_info = runner.init()
            print("Model initialized:", model_info)

            print("Listening for screams...")
            while not stop_event.ready():
                if not audio_classifier.audio_queue.empty():
                    scores = audio_classifier.classify_audio(runner)
                    print(scores)
                eventlet.sleep(0.1)  # Prevent CPU overload
        except Exception as e:
            print(f"Error: {e}")
        finally:
            runner.stop()

def emit_video_frames():
    """Capture video frames, compress them, and send them to the client."""
    model_path = 'yolo11x-pose.pt'
    fall_system = FallDetectionSystem(model_path)
    vc = cv2.VideoCapture(0)
    if not vc.isOpened():
        print("Error: Could not open video stream.")
        return

    while not stop_event.ready():
        rval, frame = vc.read()
        if not rval:
            break

        # Resize frame for compression (reduce resolution)
        if compressFrame:
            frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)

        processed_frame = fall_system.process_frame(frame)

        with lock:
            _, encoded_image = cv2.imencode(".jpg", processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            video_frames_queue.append(encoded_image.tobytes())

        eventlet.sleep(0.001)  # Maintain 30 FPS

    vc.release()

def emit_data():
    """Emit video and audio data to the client."""
    framerate = 1 / 100000000000
    while not stop_event.ready():
        with lock:
            if video_frames_queue and audio_datas_queue:
                socketio.emit('video_frame', {'frame': video_frames_queue[-1], 'audio_data': audio_datas_queue[-1]})
                video_frames_queue.popleft()
                audio_datas_queue.popleft()
                print(len(video_frames_queue))
        eventlet.sleep(framerate)

if __name__ == '__main__':
    try:
        eventlet.spawn(capture_audio)
        eventlet.spawn(emit_video_frames)
        eventlet.spawn(emit_data)

        socketio.run(app, host="0.0.0.0", port=8000, debug=False, use_reloader=False)

    except KeyboardInterrupt:
        print("\nCtrl+C detected! Stopping all threads...")
        stop_event.send()
        eventlet.sleep(1)  # Allow threads to exit gracefully
        print("Shutdown complete.")
