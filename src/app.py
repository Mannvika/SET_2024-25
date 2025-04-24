from gevent import monkey
monkey.patch_all(thread = False, select = True, dns=True, aggressive=True)

from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
import cv2
import gevent
from fall_detection_system import FallDetectionSystem
import time
import traceback
from gevent.queue import Queue
import pyaudio
import serial

# Edge Impulse Audio
from edge_impulse_linux.audio import AudioImpulseRunner

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, async_mode="gevent", cors_allowed_origins="*", allow_upgrades=False)

# Buffers
video_frames_queue = Queue(maxsize=25)
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
        'dance': 'D'
    }
    
    if data['action'] in command_map:
        state = 1 if data['state'] else 0
        direction_queue.put(f"{command_map[data['action']]}:{state}")

def arduino_writer():
    arduino = serial.Serial(
        port="/dev/ttyACM0",
        baudrate=115200,
        timeout=0.1,  # Non-blocking read
        write_timeout=0.1,  # Non-blocking write
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE)
    
    while should_run:
        try:
            while not direction_queue.empty():
                cmd = direction_queue.get_nowait()
                print(cmd)
                arduino.write(f"{cmd}\n".encode('utf-8'))  # Yields automatically
            gevent.sleep(0.0001)
        except (serial.SerialException, gevent.timeout.Timeout) as e:
            print(f"Non-blocking error: {str(e)}")
            gevent.sleep(0.1)
        finally:
            if not should_run:
                arduino.close()

   

# Replace process_audio with official generator pattern
def audio_classification_loop():
    with AudioImpulseRunner(MODEL_PATH) as runner:
        try:
            # Initialize the EdgeImpulse model was runner to be used.
            model_info = runner.init()
            
            # Get the labels Screaming and notScreaming
            labels = model_info['model_parameters']['labels']

            # Get additional model parameters including audio window to intake.
            window_size = model_info['model_parameters']['input_features_count']

            # Model information
            print(f"Loaded model: {model_info['project']['owner']}/{model_info['project']['name']}")
            print(f"Window: {window_size} samples ({window_size/MODEL_SAMPLE_RATE:.2f}s)")
            for res, audio in runner.classifier(device_id=device_id):
                # Prints how long it took to get the classification                
                print('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='')
                for label in labels:
                    # We only care about the Screaming label
                    if label == "Screaming":
                        # Gets how much the model thinks is screaming and prints it
                        score = res['result']['classification'][label]
                        print('%s: %.2f\t' % (label, score), end='')
                        # We've (Sunny, Sarah, and Antonio) determined that Screaming > 0.70
                        # is high enough confident to be consistent with a screaming sound. 
                        # Inserts a Boolean into the result of whether a scream was detected or not. 
                        if score >= 0.70:
                            print("SCREAMING")
                            result_queue.put(True)
                        else:
                            print("NOT SCREAMING")
                            result_queue.put(False)
                if not should_run:
                    break
                gevent.sleep(0)  # ← Explicit yield
        except Exception as e:
            traceback.print_exc()



def emit_video_frames():
    """Video processing pipeline"""
    model_path = 'yolo11x-pose.pt'
    fall_system = FallDetectionSystem(model_path)
    
    # Reduce frame resolution to lower processing load
    frame_width = 320
    frame_height = 240

    target_fps = 24
    frame_interval = 1.0 / target_fps

    while should_run:
        start_time = time.monotonic()
        try:
            vc = cv2.VideoCapture(0)
            if not vc.isOpened():
                print("Could not open video stream, retrying in 3 seconds...")
                gevent.sleep(3)
                continue
                
            # Set camera properties to reduce load

            while should_run:
                rval, frame = vc.read()

                if not rval:
                    break

                frame = cv2.flip(frame, -1)
                    
                # Skip frames if queue is getting full
                if video_frames_queue.qsize() > video_frames_queue.maxsize/2:
                    gevent.sleep(0.02)
                    continue
                    
                processed_frame = fall_system.process_frame(frame)
                
                # Use software encoding, not hardware encoding
                _, encoded_image = cv2.imencode(".jpg", processed_frame, 
                                              [cv2.IMWRITE_JPEG_QUALITY, 40])
                                              
                video_frames_queue.put(encoded_image.tobytes())
                elapsed = time.monotonic() - start_time
                sleep_duration = max(0, frame_interval - elapsed)
                gevent.sleep(sleep_duration)
                
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
                
            #elif not audio_queue.empty():
                #socketio.emit('audio_data', {'chunk': audio_queue.get_nowait()})
                #gevent.sleep(0.001)
                
            if not result_queue.empty():
                #This will return an queue of Booleans of whether the classification is Screaming or not.
                socketio.emit('audio_classification', {'result': result_queue.get_nowait()})
                
            gevent.sleep(0.005)
            #time.sleep(max(0.01, 1 / (2 * len(video_frames_queue) + 1)))

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
    gevent.sleep(0.1)
    
    # Find and kill all active greenlets
    try:
        greenlets = [g for g in gevent.get_hub().threadpool if not g.dead]
        gevent.killall(greenlets, timeout=3)
    except Exception as e:
        print(f"Error killing greenlets: {str(e)}")
    
    print("Shutdown complete.")

if __name__ == '__main__':
    try:
        p = pyaudio.PyAudio()
        try:
            devices = []
            for i in range(p.get_device_count()):
                dev = p.get_device_info_by_index(i)
                print(f"[{i}] {dev['name']} {dev['maxInputChannels']}")
        finally:
            p.terminate()

        device_id = int(input("Enter Device ID: "))

        gevent.spawn(arduino_writer)
        gevent.spawn(emit_data)
        gevent.spawn(emit_video_frames)
        gevent.spawn(audio_classification_loop)

        socketio.run(app, host="0.0.0.0", port=8000, debug=False)
    except KeyboardInterrupt:
        graceful_shutdown()

