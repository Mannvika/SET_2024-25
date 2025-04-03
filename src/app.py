from gevent import monkey
monkey.patch_all(thread=False, select=False)

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
import pyaudio

# Edge Impulse Audio
from edge_impulse_linux.audio import AudioImpulseRunner
from edge_impulse_linux.audio import Microphone

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, async_mode="gevent", cors_allowed_origins="*")

# Buffers
video_frames_queue = Queue(maxsize=10)
result_queue = Queue(maxsize=10)

# Audio Config
MODEL_SAMPLE_RATE = 16000
CAPTURE_SAMPLE_RATE = 44100
CHUNK_SIZE = int(MODEL_SAMPLE_RATE * 0.1)
OVERLAP = 0.25

device_id = 0
MODEL_PATH = "/home/ufset/Desktop/SET_2024-25/src/audio_model.eim"

# System State
compressFrame = False
should_run = True

class PatchedMicrophone(Microphone):
    def __init__(self, rate, chunk_size, device_id=None, channels=2):
        super().__init__(rate, chunk_size, device_id)
        self.channels = channels  # Override parent's channels=1

    def __enter__(self):
        return self  # Required for context manager

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def _init_pyaudio(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=self.device_id,
            stream_callback=self._callback
        )

class PatchedAudioImpulseRunner(AudioImpulseRunner):
    def classifier(self, device_id=None):
        return PatchedMicrophone(
            rate=self.sampling_rate,
            chunk_size=256,
            device_id=device_id,
            channels=2
        )

def audio_classification_loop():
    try:
        with runner.classifier(device_id=device_id) as mic:
            generator = mic.generator()
            features = np.array([], dtype=np.float32)
            
            while should_run:
                for audio in generator:
                    # Convert stereo to mono
                    data = np.frombuffer(audio, dtype=np.int16)
                    mono_data = data.reshape(-1, 2).mean(axis=1).astype(np.float32)
                    
                    features = np.concatenate((features, mono_data))
                    
                    while len(features) >= runner.window_size:
                        try:
                            res = runner.classify(features[:runner.window_size])
                            result_queue.put(res)
                        except Exception as e:
                            print(f"Classification error: {str(e)}")
                            traceback.print_exc()
                        
                        features = features[int(runner.window_size * OVERLAP):]
                        gevent.sleep(0)
    except Exception as e:
        print(f"Audio loop failed: {str(e)}")
        traceback.print_exc()

def emit_video_frames():
    """Video processing pipeline"""
    model_path = 'yolo11x-pose.pt'
    fall_system = FallDetectionSystem(model_path)
    
    frame_width = 320
    frame_height = 240
    frame_rate = 15

    while should_run:
        try:
            vc = cv2.VideoCapture(0)
            if not vc.isOpened():
                print("Could not open video stream, retrying...")
                gevent.sleep(3)
                continue
                
            vc.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
            vc.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
            vc.set(cv2.CAP_PROP_FPS, frame_rate)

            while should_run:
                rval, frame = vc.read()
                if not rval:
                    break
                    
                if video_frames_queue.qsize() > 5:
                    gevent.sleep(0.2)
                    continue
                    
                processed_frame = fall_system.process_frame(frame)
                _, encoded_image = cv2.imencode(".jpg", processed_frame, 
                                              [cv2.IMWRITE_JPEG_QUALITY, 40])
                video_frames_queue.put(encoded_image.tobytes())
                gevent.sleep(1 / frame_rate)
                
        except Exception as e:
            print(f"Video error: {str(e)}")
            traceback.print_exc()
        finally:
            if 'vc' in locals():
                vc.release()
            print("Video device released")
            gevent.sleep(2)

def emit_data():
    """Data emitter with enhanced error handling"""
    while should_run:
        try:
            if not video_frames_queue.empty():
                socketio.emit('video_frame', {
                    'frame': video_frames_queue.get_nowait()
                })
            
            if not result_queue.empty():
                res = result_queue.get_nowait()
                socketio.emit('audio_classification', {
                    'result': res['result']['classification']
                })

            gevent.sleep(0.001)
        except Exception as e:
            print(f"Emit error: {str(e)}")
            traceback.print_exc()

def graceful_shutdown():
    global should_run
    should_run = False
    gevent.sleep(1)
    
    if 'runner' in globals():
        try:
            runner.stop()
        except Exception as e:
            print(f"Runner stop error: {str(e)}")
    
    try:
        greenlets = [g for g in gevent.get_hub().threadpool if not g.dead]
        gevent.killall(greenlets, timeout=5)
    except Exception as e:
        print(f"Greenlet kill error: {str(e)}")
    
    print("Shutdown complete")

if __name__ == '__main__':
    try:
        global runner, labels, window_size
        
        runner = PatchedAudioImpulseRunner(MODEL_PATH)
        model_info = runner.init()
        
        labels = model_info['model_parameters']['labels']
        window_size = model_info['model_parameters']['input_features_count']

        print(f"Loaded model: {model_info['project']['owner']}/{model_info['project']['name']}")
        print(f"Window size: {window_size} samples")

        p = pyaudio.PyAudio()
        print("Available PyAudio devices:")
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            print(f"{i}: {dev['name']} (Input channels: {dev['maxInputChannels']})")
        
        device_id = int(input("Enter PyAudio device ID: "))

        gevent.spawn(audio_classification_loop)
        gevent.spawn(emit_video_frames)
        gevent.spawn(emit_data)

        socketio.run(app, host="0.0.0.0", port=8000, debug=False)

    except KeyboardInterrupt:
        graceful_shutdown()
    except Exception as e:
        print(f"Main error: {str(e)}")
        traceback.print_exc()
        graceful_shutdown()
