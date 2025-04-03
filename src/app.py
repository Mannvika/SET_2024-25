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
        self.actual_sample_rate = None
        super().__init__(rate, chunk_size, device_id)
        self.channels = channels
        self.resample_required = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def _init_pyaudio(self):
        self.p = pyaudio.PyAudio()
        
        # Get device info
        dev_info = self.p.get_device_info_by_index(self.device_id)
        self.actual_sample_rate = int(dev_info['defaultSampleRate'])
        self.resample_required = (self.actual_sample_rate != MODEL_SAMPLE_RATE)
        
        print(f"Device sample rate: {self.actual_sample_rate}Hz, Model requires: {MODEL_SAMPLE_RATE}Hz")
        
        # Force format check
        try:
            is_supported = self.p.is_format_supported(
                rate=MODEL_SAMPLE_RATE,
                input_device=self.device_id,
                input_channels=1,
                input_format=pyaudio.paInt16
            )
            print(f"Format supported: {is_supported}")
        except Exception as e:
            print(f"Compatibility check failed: {str(e)}")
            raise

        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.actual_sample_rate if self.resample_required else MODEL_SAMPLE_RATE,
            input=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=self.device_id,
            stream_callback=self._callback
        )

    def generator(self):
        for raw_audio in super().generator():
            if self.resample_required:
                # Convert to numpy array and resample
                data = np.frombuffer(raw_audio, dtype=np.int16)
                data = librosa.resample(
                    data.astype(np.float32),
                    orig_sr=self.actual_sample_rate,
                    target_sr=MODEL_SAMPLE_RATE
                ).astype(np.int16).tobytes()
            yield data

class PatchedAudioImpulseRunner(AudioImpulseRunner):
    def classifier(self, device_id=None):
        return PatchedMicrophone(
            rate=MODEL_SAMPLE_RATE,
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
        global runner

        runner = PatchedAudioImpulseRunner(MODEL_PATH)
        model_info = runner.init()

        print(f"Loaded model: {model_info['project']['owner']}/{model_info['project']['name']}")
        print(f"Window size: {model_info['model_parameters']['input_features_count']} samples")

        p = pyaudio.PyAudio()
        print("\nAvailable PyAudio devices:")
        valid_devices = []
        
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            channels = dev['maxInputChannels']
            rate = int(dev['defaultSampleRate'])
            
            print(f"{i}: {dev['name']}")
            print(f"   Channels: {channels}, Sample Rate: {rate}Hz")
            
            if channels >= 1 and (rate == MODEL_SAMPLE_RATE or rate >= 16000):
                valid_devices.append(i)
                print("   → VALID DEVICE")

        if not valid_devices:
            raise Exception("No compatible audio devices found!")

        device_id = int(input(f"\nEnter VALID PyAudio device ID ({valid_devices}): "))
        if device_id not in valid_devices:
            raise ValueError("Invalid device selected")

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
