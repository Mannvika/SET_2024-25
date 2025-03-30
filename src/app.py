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
audio_datas_queue = deque()
SAMPLE_RATE = 44100
CHUNK_SIZE = 512
starttime = time.time()

compressFrame = False
device_id = 0  # Change if needed

MODEL_PATH = "/home/ufset/Desktop/SET_2024-25/src/audio_model.eim"

def capture_audio():
    """Capture audio in real-time and send to the client."""


    with AudioImpulseRunner(MODEL_PATH) as runner:
        try:
            model_info = runner.init()
            labels = model_info['model_parameters']['labels']
            print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')

            for res, audio in runner.classifier(device_id=selected_device_id):
                print('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='')
                for label in labels:
                    score = res['result']['classification'][label]
                    print('%s: %.2f\t' % (label, score), end='')
                print('', flush=True)
        finally:
            if (runner):
                runner.stop()


    '''
    audio_classifier = AudioClassifier(device_id)

    def audio_callback(indata, frames, time, status):
        if status:
            print(status)

        #print(indata.tolist())
        #audio_datas_queue.append(indata.tolist())

    runner = AudioImpulseRunner(MODEL_PATH)
    while True:
        scores = audio_classifier.classify_audio(runner)
        print(scores)
        gevent.sleep(5)
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=2, callback=audio_callback, blocksize=CHUNK_SIZE, device=device_id, latency='low'):
        try:
            model_info = runner.init()
            print("Model initialized:", model_info)

            print("Listening for screams...")
            while True:
                if not audio_classifier.audio_queue.empty():
                    scores = audio_classifier.classify_audio(runner)
                    print(scores)
                gevent.sleep(1)  # Prevent CPU overload
        except Exception as e:
            print(f"Error: {e}")
        finally:
            runner.stop()
    '''

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
        if video_frames_queue and audio_datas_queue and (time.time() - starttime > 10):
            socketio.emit('video_frame', {'frame': video_frames_queue[-1], 'audio_data': audio_datas_queue[-1]})
            video_frames_queue.popleft()
            audio_datas_queue.popleft()
            print(len(video_frames_queue))
        gevent.sleep(0.001)

if __name__ == '__main__':
    try:
        print(sd.query_devices())
        id = int(input("Enter Device ID: "))
        device_id = id

        starttime = time.time()
        gevent.spawn(capture_audio)
        gevent.spawn(emit_video_frames)
        gevent.spawn(emit_data)

        socketio.run(app, host="0.0.0.0", port=8000, debug=False, use_reloader=False)

    except KeyboardInterrupt:
        print("\nCtrl+C detected! Stopping...")
        gevent.sleep(1)  # Allow graceful shutdown
        print("Shutdown complete.")
