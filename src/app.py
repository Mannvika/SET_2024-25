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

# Add these new functions to handle audio processing
def audio_capture():
    """Continuous audio capture using sounddevice"""
    try:
        with sd.InputStream(samplerate=CAPTURE_SAMPLE_RATE, 
                          channels=2,
                          device=device_id,
                          blocksize=CHUNK_SIZE,
                          callback=audio_callback):
            while should_run:
                gevent.sleep(0.1)
    except Exception as e:
        print(f"Audio capture failed: {str(e)}")

# Modified audio callback for stereo conversion
def audio_callback(indata, frames, time, status):
    if status:
        print(f"Audio status: {status}")
    
    # Convert stereo to mono by averaging channels
    mono_data = np.mean(indata, axis=1)
    
    # Resample to model's sample rate
    chunk = librosa.resample(mono_data,
                           orig_sr=CAPTURE_SAMPLE_RATE,
                           target_sr=MODEL_SAMPLE_RATE)
    
    audio_queue.put(chunk.astype(np.float32))

def process_audio():
    """Audio classification worker using Edge Impulse"""
    try:
        pool = ThreadPool(2)
        features = []
        
        while should_run:
            if audio_queue.empty():
                gevent.sleep(0.01)
                continue
                
            chunk = audio_queue.get()
            features.extend(chunk)
            
            # Maintain sliding window with overlap
            while len(features) >= window_size:
                pool.apply_async(classify_audio, 
                               args=(features[:window_size],))
                features = features[int(window_size*(1-OVERLAP)):]
                
    except Exception as e:
        print(f"Audio processing error: {str(e)}")

def classify_audio(data):
    """Edge Impulse classification in threadpool"""
    try:
        res = runner.classify(np.array(data))
        classification_queue.put(res)
    except Exception as e:
        print(f"Classification error: {str(e)}")

def emit_video_frames():
    """Video processing pipeline"""
    model_path = 'yolo11x-pose.pt'
    fall_system = FallDetectionSystem(model_path)
    
    # Reduce frame resolution to lower processing load
    frame_width = 320
    frame_height = 240
    frame_rate = 15  # Reduced from 30 fps

    while should_run:
        try:
            vc = cv2.VideoCapture(0)
            if not vc.isOpened():
                print("Could not open video stream, retrying in 3 seconds...")
                gevent.sleep(3)
                continue
                
            # Set camera properties to reduce load
            vc.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
            vc.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
            vc.set(cv2.CAP_PROP_FPS, frame_rate)

            while should_run:
                rval, frame = vc.read()
                if not rval:
                    break
                    
                # Skip frames if queue is getting full
                if video_frames_queue.qsize() > 30:
                    gevent.sleep(0.1)
                    continue
                    
                processed_frame = fall_system.process_frame(frame)
                
                # Use software encoding, not hardware encoding
                _, encoded_image = cv2.imencode(".jpg", processed_frame, 
                                              [cv2.IMWRITE_JPEG_QUALITY, 40])
                                              
                video_frames_queue.put(encoded_image.tobytes())
                gevent.sleep(1 / frame_rate)
                
        except Exception as e:
            print(f"Video capture error: {str(e)}")
        finally:
            if 'vc' in locals() and vc is not None:
                vc.release()
            print("Video device released, will attempt reconnection")
            gevent.sleep(2)

def emit_data():
    """Unified data emitter with error handling"""
    try:
        if not video_frames_queue.empty():
            socketio.emit('video_frame', {'frame': video_frames_queue.get_nowait()})
        
        if not audio_queue.empty():
            socketio.emit('audio_data', {'chunk': audio_queue.get_nowait().tobytes()})
        
        if not classification_queue.empty():
            res = classification_queue.get_nowait()
            result = {
                'label': max(res['result']['classification'], 
                            key=res['result']['classification'].get),
                'scores': res['result']['classification']
            }
            socketio.emit('audio_classification', result)

        gevent.sleep(0.001)

    except BrokenPipeError:
        print("Client disconnected - resetting queues")
    
    except Exception as e:
        print(f"Emit error: {str(e)}")


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

        gevent.spawn(emit_data)
        gevent.spawn(emit_video_frames)
        gevent.spawn(audio_capture)
        gevent.spawn(process_audio)

        socketio.run(app, host="0.0.0.0", port=8000, debug=False)

    except KeyboardInterrupt:
        graceful_shutdown()
