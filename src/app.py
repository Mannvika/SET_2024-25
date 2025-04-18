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
from gevent.threadpool import ThreadPool
import librosa
import psutil  # For resource monitoring
import pyaudio
import serial

# Edge Impulse Audio
from edge_impulse_linux.audio import AudioImpulseRunner

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, async_mode="gevent", cors_allowed_origins="*")

# Buffers
serial_pool = ThreadPool(1)
video_frames_queue = Queue(maxsize=10)
audio_queue = Queue(maxsize=5)
result_queue = Queue(maxsize=10)
direction_queue = Queue(maxsize=20)

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

@socketio.on('movement_command')
def handle_direction(data):
    command_map = {
        'forward': 'F', 
        'backward': 'B',
        'left': 'L',
        'right': 'R',
        'turn': 'T'
    }
    
    if data['action'] in command_map:
        state = 1 if data['state'] else 0
        direction_queue.put(f"{command_map[data['action']]}:{state}")

def serial_worker():
    """Dedicated OS thread for blocking serial operations"""
    arduino = None
    while should_run:
        try:
            if not arduino:
                arduino = serial.Serial(
                    port='/dev/ttyACM0',
                    baudrate=115200,
                    timeout=0.1  # Critical for non-blocking reads
                )
                sleep(2)  # Allow Arduino reset

            # Non-blocking write
            if not command_queue.empty():
                cmd = command_queue.get_nowait()
                arduino.write(f"{cmd}\n".encode('utf-8'))

            # Non-blocking read (optional)
            while arduino.in_waiting > 0:
                data = arduino.read_all()
                process_serial_data(data)

        except (serial.SerialException, OSError) as e:
            print(f"Serial error: {e}")
            if arduino:
                arduino.close()
                arduino = None
            sleep(1)

def arduino_writer():
    """Gevent-compatible queue manager"""
    while should_run:
        # Yield control to other greenlets
        sleep(0)


# Replace process_audio with official generator pattern
def audio_classification_loop():
    try:
        runner = AudioImpulseRunner(MODEL_PATH)
        model_info = runner.init()
        
        labels = model_info['model_parameters']['labels']
        window_size = model_info['model_parameters']['input_features_count']

        print(f"Loaded model: {model_info['project']['owner']}/{model_info['project']['name']}")
        print(f"Window: {window_size} samples ({window_size/MODEL_SAMPLE_RATE:.2f}s)")
        for res, audio in runner.classifier(device_id=device_id):
            print("bello")
            if not should_run:
                break
            result_queue.put(res)
            gevent.sleep(0)  # ← Explicit yield
        print("bello again")
    except Exception as e:
        traceback.print_exc()


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
                if video_frames_queue.qsize() > 5:
                    gevent.sleep(0.2)
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
        
        if not result_queue.empty():
            socketio.emit('audio_classification', {'result': result_queue.get_nowait()})

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
        #global runner, labels, window_size
        
        #runner = AudioImpulseRunner(MODEL_PATH)
        #model_info = runner.init()
        
        #labels = model_info['model_parameters']['labels']
        #window_size = model_info['model_parameters']['input_features_count']

        #print(f"Loaded model: {model_info['project']['owner']}/{model_info['project']['name']}")
        #print(f"Window: {window_size} samples ({window_size/MODEL_SAMPLE_RATE:.2f}s)")

        p = pyaudio.PyAudio()
        try:
            devices = []
            for i in range(p.get_device_count()):
                dev = p.get_device_info_by_index(i)
                print(f"[{i}] {dev['name']} {dev['maxInputChannels']}")
        finally:
            p.terminate()        
            
        device_id = int(input("Enter Device ID: "))

        gevent.spawn(emit_data)
        gevent.spawn(emit_video_frames)
        #gevent.spawn(audio_classification_loop)
        gevent.spawn(arduino_writer)

        socketio.run(app, host="0.0.0.0", port=8000, debug=False)

    except KeyboardInterrupt:
        graceful_shutdown()
