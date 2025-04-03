from gevent import monkey
monkey.patch_all(thread=False, select=False)

from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
import cv2
import gevent
import numpy as np
from fall_detection_system import FallDetectionSystem
import traceback
from gevent.queue import Queue
import pyaudio
import time

# Edge Impulse Audio
from edge_impulse_linux.audio import AudioImpulseRunner

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, async_mode="gevent", cors_allowed_origins="*")

# Buffers
video_frames_queue = Queue(maxsize=10)
audio_results_queue = Queue(maxsize=10)

# Configuration
AUDIO_MODEL_PATH = "/home/ufset/Desktop/SET_2024-25/src/audio_model.eim"
VIDEO_FRAME_RATE = 15
VIDEO_RESOLUTION = (320, 240)

# System State
should_run = True

def safe_audio_classification(device_id):
    """Audio processing with isolated error handling"""
    runner = None
    try:
        with AudioImpulseRunner(AUDIO_MODEL_PATH) as runner:
            model_info = runner.init()
            print(f"Audio model loaded: {model_info['project']['name']}")
            
            for res, _ in runner.classifier(device_id=device_id):
                if not should_run:
                    break
                audio_results_queue.put({
                    'classification': res['result']['classification'],
                    'timing': res['timing']
                })

    except Exception as e:
        print(f"Audio thread crashed: {str(e)}")
        traceback.print_exc()
    finally:
        if runner:
            runner.stop()
        print("Audio processing stopped")

def emit_video_frames():
    """Video processing with fall detection"""
    fall_system = FallDetectionSystem('yolo11x-pose.pt')
    
    while should_run:
        try:
            with cv2.VideoCapture(0) as vc:
                if not vc.isOpened():
                    raise RuntimeError("Video capture failed")
                
                vc.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_RESOLUTION[0])
                vc.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_RESOLUTION[1])
                vc.set(cv2.CAP_PROP_FPS, VIDEO_FRAME_RATE)

                while should_run:
                    ret, frame = vc.read()
                    if not ret:
                        break
                    
                    processed_frame = fall_system.process_frame(frame)
                    _, encoded = cv2.imencode(".jpg", processed_frame, 
                                           [cv2.IMWRITE_JPEG_QUALITY, 40])
                    video_frames_queue.put(encoded.tobytes())
                    gevent.sleep(1/VIDEO_FRAME_RATE)

        except Exception as e:
            print(f"Video error: {str(e)}")
            traceback.print_exc()
            gevent.sleep(2)

def emit_data():
    """Safe data emission handler"""
    while should_run:
        try:
            # Process video frames
            if not video_frames_queue.empty():
                socketio.emit('video_frame', {
                    'frame': video_frames_queue.get_nowait()
                })

            # Process audio results
            if not audio_results_queue.empty():
                res = audio_results_queue.get_nowait()
                socketio.emit('audio_classification', {
                    'result': res['classification']
                })

            gevent.sleep(0.001)
        except Exception as e:
            print(f"Emission error: {str(e)}")
            traceback.print_exc()

def graceful_shutdown():
    global should_run
    should_run = False
    gevent.sleep(1)
    print("System shutdown complete")

def get_valid_audio_device():
    """Safe device selection with validation"""
    p = pyaudio.PyAudio()
    try:
        devices = []
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev['maxInputChannels'] > 0:
                devices.append((i, dev['name']))
        
        print("Available audio devices:")
        for idx, name in devices:
            print(f"[{idx}] {name}")
        
        while True:
            try:
                device_id = int(input("Enter audio device ID: "))
                if any(idx == device_id for idx, _ in devices):
                    return device_id
                print("Invalid device ID, try again")
            except ValueError:
                print("Please enter a numeric device ID")
    finally:
        p.terminate()

if __name__ == '__main__':
    try:
        # Start video and emission threads
        gevent.spawn(emit_video_frames)
        gevent.spawn(emit_data)
        
        # Start audio thread with isolated error handling
        audio_device = get_valid_audio_device()
        gevent.spawn(safe_audio_classification, audio_device)

        # Start server
        socketio.run(app, host="0.0.0.0", port=8000, debug=False)

    except KeyboardInterrupt:
        graceful_shutdown()
    except Exception as e:
        print(f"Main error: {str(e)}")
        traceback.print_exc()
        graceful_shutdown()
