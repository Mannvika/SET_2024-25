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

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, async_mode="gevent", cors_allowed_origins="*")

# Buffers
video_frames_queue = Queue(maxsize=10)
result_queue = Queue(maxsize=10)

# Audio Configuration
MODEL_PATH = "/home/ufset/Desktop/SET_2024-25/src/audio_model.eim"
CHUNK_SIZE = 1024
OVERLAP = 0.25

# System State
compressFrame = False
should_run = True

# --- Audio Classification Thread ---
def audio_classification_thread(device_id):
    """Main loop for audio classification"""
    try:
        with AudioImpulseRunner(MODEL_PATH) as runner:
            model_info = runner.init()
            print(f"Audio model: {model_info['project']['owner']}/{model_info['project']['name']}")
            
            for res, audio in runner.classifier(device_id=device_id):
                if not should_run:
                    break
                
                # Put result in queue for emission
                result_queue.put_nowait({
                    'result': res['result']['classification'],
                    'timing': res['timing']
                })
                
    except Exception as e:
        print(f"Audio classification error: {str(e)}")
        traceback.print_exc()
    finally:
        print("Audio classification stopped")

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
        # Audio Device Selection
        p = pyaudio.PyAudio()
        print("\nAvailable PyAudio devices:")
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev['maxInputChannels'] > 0:
                print(f"{i}: {dev['name']} (Channels: {dev['maxInputChannels']}, SR: {dev['defaultSampleRate']}Hz)")
        device_id = int(input("Enter valid audio device ID: "))
        p.terminate()

        # Start services
        gevent.spawn(emit_video_frames)
        gevent.spawn(emit_data)
        gevent.spawn(audio_classification_thread, device_id)  # Add audio thread

        # Start server
        socketio.run(app, host="0.0.0.0", port=8000, debug=False)

    except KeyboardInterrupt:
        graceful_shutdown()
    except Exception as e:
        print(f"Main error: {str(e)}")
        traceback.print_exc()
        graceful_shutdown()
