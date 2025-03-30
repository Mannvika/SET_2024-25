from gevent import monkey
monkey.patch_all()

from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
import cv2
import sounddevice as sd
import gevent
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
socketio = SocketIO(app, async_mode="gevent", cors_allowed_origins="*")

video_frames_queue = deque()
audio_queue = deque()
classification_queue = deque()
result_queue = deque()
#SAMPLE_RATE = 44100
CHUNK_SIZE = 1024
OVERLAP = 0.25
starttime = time.time()

compressFrame = False
device_id = 0  # Change if needed

MODEL_PATH = "/home/ufset/Desktop/SET_2024-25/src/audio_model.eim"

def capture_audio():
    """Capture audio in real-time and send to the client."""
    #audio_classifier = AudioClassifier(device_id)

    while True:
        # Capture audio data in chunks
        audio_data = sd.rec(CHUNK_SIZE, samplerate=sampling_rate, channels=2, dtype='int16')
        sd.wait()  # Wait until the recording is finished
        audio_queue.put(audio_data)  # Put the captured audio into the queue
        classification_queue.put(audio_data)
        gevent.sleep(0.5)  # Adjust based on your real-time performance needs

def classify_audio():
    """Classify audio data in real-time."""
    features = np.array([], dtype=np.int16)  # Buffer to hold audio data
    while True:
        if not audio_queue.empty():
            # Get the latest chunk of audio from the queue
            audio_data = classification_queue.get()

            # Add the new audio data to the buffer
            features = np.concatenate((features, audio_data), axis=0)

            # Check if we have enough data to classify
            while len(features) >= window_size:
                # Extract a window of audio data for classification
                window = features[:window_size]

                # Classify the window of audio data
                res = runner.classify(window.tolist())

                result_queue.append(res)

                # Output the results
                print(f"Result: {res[0]['result']}")  # Assuming classify returns a list of results
                for label in labels:
                    score = res[0]['result']['classification'][label]
                    print(f"{label}: {score:.2f}")

                # Remove the processed window from the buffer
                features = features[int(window_size * (1 - OVERLAP)):]

            gevent.sleep(0.5)

def emit_video_frames():
    """Capture video frames, compress them, and send them to the client."""
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

        if compressFrame:
            frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)

        processed_frame = fall_system.process_frame(frame)
        _, encoded_image = cv2.imencode(".jpg", processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        video_frames_queue.append(encoded_image.tobytes())

        gevent.sleep(0.001)  # Maintain 30 FPS

    vc.release()

def emit_data():
    """Emit video and audio data to the client."""
    while True:
        if video_frames_queue and (time.time() - starttime > 10):
            socketio.emit('video_frame', {'frame': video_frames_queue[-1]})
            video_frames_queue.popleft()
        
        if result_queue:
            classification_result = result_queue.popleft()
            socketio.emit('audio_classification', {'result': classification_result})

        if audio_queue:
            socketio.emit('audio_data', {'chunk': audio_queue[-1]})
            audio_queue.popleft()

        gevent.sleep(0.01)  # Short sleep for responsivenes

if __name__ == '__main__':
    try:
        print(sd.query_devices())
        id = int(input("Enter Device ID: "))
        device_id = id

        global runner, labels, window_size, sampling_rate
        runner = AudioImpulseRunner(MODEL_PATH)
        model_info = runner.init()                
        labels = model_info['model_parameters']['labels']
        window_size = model_info['model_parameters']['input_features_count']
        sampling_rate = model_info['model_parameters']['frequency']
        print(f"Loaded model: {model_info['project']['owner']} / {model_info['project']['name']}")

        starttime = time.time()
        gevent.spawn(capture_audio)
        gevent.spawn(classify_audio)
        gevent.spawn(emit_video_frames)
        gevent.spawn(emit_data)

        socketio.run(app, host="0.0.0.0", port=8000, debug=False, use_reloader=False)

    except KeyboardInterrupt:
        print("\nCtrl+C detected! Stopping...")
        gevent.sleep(1)  # Allow graceful shutdown
        print("Shutdown complete.")
