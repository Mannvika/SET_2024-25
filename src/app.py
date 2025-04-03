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
import traceback
from gevent.queue import Queue
from gevent.threadpool import ThreadPool
import librosa
import psutil  # For resource monitoring

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

device_id = 0  # Set device ID dynamically at runtime
MODEL_PATH = "/home/ufset/Desktop/SET_2024-25/src/audio_model.eim"

# System State
compressFrame = False
should_run = True


def audio_callback(indata, frames, time, status):
    """Audio capture callback with resampling"""
    if status:
        print("Audio error:", status)
    try:
        mono_audio = np.mean(indata, axis=1).astype(np.float32)  # Stereo to mono conversion
        resampled = librosa.resample(
            mono_audio.T,
            orig_sr=CAPTURE_SAMPLE_RATE,
            target_sr=MODEL_SAMPLE_RATE,
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
            device=device_id,
        ):
            print("Audio capture running...")
            while should_run:
                gevent.sleep(0.1)
    except Exception as e:
        print(f"Error initializing audio stream: {str(e)}")


def classify_audio():
    """Async classification using thread pool"""
    pool = ThreadPool(1)
    features = np.array([], dtype=np.float32)

    while 'window_size' not in globals() or window_size is None:
        print("Waiting for audio model initialization...")
        gevent.sleep(1)

    print(f"Starting audio classification with window size: {window_size}")

    while should_run:
        try:
            while features.shape[0] < window_size:
                chunk = classification_queue.get(timeout=1)
                features = np.concatenate((features, chunk.flatten()))

            window = features[:window_size]
            if np.isnan(window).any() or np.isinf(window).any():
                print("Invalid audio data detected, resetting buffer")
                features = np.array([], dtype=np.float32)
                continue

            future = pool.spawn(runner.classify, window.tolist())
            features = features[int(window_size * (1 - OVERLAP)):]

            def callback(f):
                try:
                    result_queue.put(f.get())
                except Exception as e:
                    print(f"Classification error details: {str(e)}")

            future.link(callback)
            gevent.sleep(0.01)

        except Exception as e:
            print(f"Classification pipeline error: {str(e)}")
            traceback_info = traceback.format_exc()
            print(f"Traceback: {traceback_info}")
            features = np.array([], dtype=np.float32)  # Reset on error
            gevent.sleep(1)


def emit_video_frames():
    """Video processing pipeline"""
    model_path = 'yolo11x-pose.pt'
    fall_system = FallDetectionSystem(model_path)

    while should_run:  # Outer loop for reconnection
        try:
            vc = cv2.VideoCapture(0)
            if not vc.isOpened():
                print("Could not open video stream, retrying in 3 seconds...")
                gevent.sleep(3)
                continue

            while should_run:
                rval, frame = vc.read()
                if not rval:
                    break

                if compressFrame:
                    frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)

                processed_frame = fall_system.process_frame(frame)
                _, encoded_image = cv2.imencode(".jpg", processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                video_frames_queue.put(encoded_image.tobytes())
                
                # Check if queue is getting full and add frame skipping
                if video_frames_queue.qsize() > 40:
                    gevent.sleep(0.1)
                    continue
                
                gevent.sleep(1 / 30)  # ~30 FPS

        except Exception as e:
            print(f"Video capture error: {str(e)}")
        finally:
            if 'vc' in locals() and vc is not None:
                vc.release()
            print("Video device released, will attempt reconnection")
            gevent.sleep(2)


def emit_data():
    """Unified data emitter with error handling"""
    while should_run:
        try:
            if not video_frames_queue.empty():
                socketio.emit('video_frame', {'frame': video_frames_queue.get_nowait()})
            
            if not audio_queue.empty():
                socketio.emit('audio_data', {'chunk': audio_queue.get_nowait().tobytes()})
            
            if not result_queue.empty():
                res = result_queue.get_nowait()
                socketio.emit('audio_classification', {'result': res['result']['classification']})

            gevent.sleep(0.001)

        except BrokenPipeError:
            print("Client disconnected - resetting queues")
        
        except Exception as e:
            print(f"Emit error: {str(e)}")


def monitor_resources():
    """Monitor system resources and adjust processing if needed"""
    global compressFrame

    while should_run:
        cpu_percent = psutil.cpu_percent(interval=1)
        mem_percent = psutil.virtual_memory().percent

        if cpu_percent > 85 or mem_percent > 80:
            compressFrame = True
            print(f"High resource usage detected (CPU: {cpu_percent}%, MEM: {mem_percent}%). Enabling compression.")
        else:
            compressFrame = False
        
        gevent.sleep(5)


def graceful_shutdown():
    """Properly handle cleanup of all resources"""
    print("\nGraceful shutdown...")
    
    # Stop all processing first
    global should_run
    should_run = False
    gevent.sleep(0.5)
    
    # Close resources
    if 'runner' in globals() and runner is not None:
        try:
            runner.stop()
        except Exception as e:
            print(f"Error stopping runner: {str(e)}")
    
    # Find and kill all active greenlets
    try:
        greenlets = [g for g in gevent.get_hub().threadpool if not g.dead]
        gevent.killall(greenlets, timeout=3)
    except Exception as e:
        print(f"Error killing greenlets: {str(e)}")
    
    print("Shutdown complete.")


if __name__ == '__main__':
    try:
        global runner, labels, window_size
        
        runner = AudioImpulseRunner(MODEL_PATH)
        model_info = runner.init()
        
        labels = model_info['model_parameters']['labels']
        window_size = model_info['model_parameters']['input_features_count']

        print(f"Loaded model: {model_info['project']['owner']}/{model_info['project']['name']}")
        print(f"Window: {window_size} samples ({window_size/MODEL_SAMPLE_RATE:.2f}s)")

        print(sd.query_devices())
        device_id = int(input("Enter Device ID: "))

        gevent.spawn(monitor_resources)
        gevent.spawn(emit_data)
        gevent.spawn(emit_video_frames)
        gevent.spawn(capture_audio)
        gevent.spawn(classify_audio)

        socketio.run(app, host="0.0.0.0", port=8000, debug=False)

    except KeyboardInterrupt:
        graceful_shutdown()
