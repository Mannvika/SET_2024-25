## app.py (Revised)
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
from gevent.queue import Queue
from gevent.threadpool import ThreadPool
import librosa

# Edge Impulse Audio
from edge_impulse_linux.audio import AudioImpulseRunner

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, async_mode="gevent", cors_allowed_origins="*")

# Buffers
video_frames_queue = Queue(maxsize=50)
audio_queue = Queue(maxsize=50)
classification_queue = Queue(maxsize=50)
result_queue = Queue(maxsize=50)

# Audio Config
MODEL_SAMPLE_RATE = 16000
CAPTURE_SAMPLE_RATE = 44100
CHUNK_SIZE = int(MODEL_SAMPLE_RATE * 0.1)  # 100ms chunks for 16kHz
OVERLAP = 0.25
#window_size = 0  # Will be set from model
device_id = 0 # Set device id

MODEL_PATH = "/home/ufset/Desktop/SET_2024-25/src/audio_model.eim"

# System State
compressFrame = False
#runner = None
#labels = []

def audio_callback(indata, frames, time, status):
    """Audio capture callback with resampling"""
    if status:
        print("Audio error:", status)
    try:
        # Stereo to mono conversion
        mono_audio = np.mean(indata, axis=1).astype(np.float32)

        # Resample to model's expected rate
        resampled = librosa.resample(
            mono_audio.T,
            orig_sr=CAPTURE_SAMPLE_RATE,
            target_sr=MODEL_SAMPLE_RATE
        ).reshape(-1, 1)
        
        classification_queue.put_nowait(resampled)
        audio_queue.put_nowait(mono_audio)
    except Exception as e:
        print(f"Audio processing error: {str(e)}")

def capture_audio():
    """Non-blocking audio capture greenlet"""
    try:
        with sd.InputStream(
            callback=audio_callback,
            channels=2,
            samplerate=CAPTURE_SAMPLE_RATE,
            dtype='float32',
            blocksize=CHUNK_SIZE,
            device=device_id
        ):
            print("Audio capture running...")
            while True:
                gevent.sleep(0.1)
    except Exception as e:
        print(f"Error initializing audio stream: {str(e)}")

def classify_audio():
    """Async classification using thread pool"""
    pool = ThreadPool(1)
    features = np.array([], dtype=np.float32)
    
    while True:
        try:
            # Build audio window
            while features.shape[0] < window_size:
                chunk = classification_queue.get(timeout=1)
                features = np.concatenate((features, chunk.flatten()))

            # Extract classification window
            window = features[:window_size]
            future = pool.spawn(runner.classify, window.tolist())
            
            # Process remaining audio with overlap
            features = features[int(window_size * (1 - OVERLAP)):]
            
            # Store result when ready
            def callback(f):
                try:
                    result_queue.put(f.get())
                except Exception as e:
                    print(f"Classification error: {str(e)}")
            
            future.link(callback)
            gevent.sleep(0)
            
        except Exception as e:
            print(f"Classification pipeline error: {str(e)}")
            gevent.sleep(1)

def emit_video_frames():
    """Video processing pipeline"""
    model_path = 'yolo11x-pose.pt'
    fall_system = FallDetectionSystem(model_path)
    
    vc = cv2.VideoCapture(0)
    if not vc.isOpened():
        raise RuntimeError("Could not open video stream")

    try:
        while True:
            rval, frame = vc.read()
            if not rval:
                break

            if compressFrame:
                frame = cv2.resize(frame, (320, 240), 
                                 interpolation=cv2.INTER_AREA)

            processed_frame = fall_system.process_frame(frame)
            _, encoded_image = cv2.imencode(".jpg", processed_frame, 
                                          [cv2.IMWRITE_JPEG_QUALITY, 50])
            video_frames_queue.put(encoded_image.tobytes())
            gevent.sleep(1/30)  # ~30 FPS
    finally:
        vc.release()

def emit_data():
    """Unified data emitter with error handling"""
    while True:
        try:
            # Video
            if not video_frames_queue.empty():
                socketio.emit('video_frame', {
                    'frame': video_frames_queue.get_nowait()
                })
            
            # Audio
            if not audio_queue.empty():
                socketio.emit('audio_data', {
                    'chunk': audio_queue.get_nowait().tobytes()
                })
            
            # Classification results
            if not result_queue.empty():
                res = result_queue.get_nowait()
                socketio.emit('audio_classification', {
                    'result': res['result']['classification']
                })
            
            gevent.sleep(0.001)
            
        except BrokenPipeError:
            print("Client disconnected - resetting queues")
            while not video_frames_queue.empty():
                video_frames_queue.get_nowait()
            while not result_queue.empty():
                result_queue.get_nowait()
        except Exception as e:
            print(f"Emit error: {str(e)}")

if __name__ == '__main__':
    try:
        # Audio model initialization
        global runner, labels, window_size
        MODEL_PATH = "/home/ufset/Desktop/SET_2024-25/src/audio_model.eim"
        runner = AudioImpulseRunner(MODEL_PATH)
        model_info = runner.init()
        labels = model_info['model_parameters']['labels']
        window_size = model_info['model_parameters']['input_features_count']
        
        print(f"Loaded model: {model_info['project']['owner']}/"
              f"{model_info['project']['name']}")
        print(f"Window: {window_size} samples "
              f"({window_size/MODEL_SAMPLE_RATE:.2f}s)")

        #Device ID prompt
        print(sd.query_devices())
        id = int(input("Enter Device ID: "))
        device_id = id

        # Start greenlets in priority order
        gevent.spawn(emit_data)
        gevent.spawn(emit_video_frames)
        gevent.spawn(capture_audio)
        gevent.spawn(classify_audio)

        socketio.run(app, host="0.0.0.0", port=8000, debug=False)

    except KeyboardInterrupt:
        print("\nGraceful shutdown...")
        runner.stop()
        # Get all active greenlets
        greenlets = [
            g for g in gevent.get_hub().threadpool 
            if not g.dead
        ]
    
        gevent.killall(greenlets, timeout=3)
        gevent.sleep(1)
        print("Shutdown complete.")
