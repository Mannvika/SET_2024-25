from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
import signal
import sys
import cv2
import multiprocessing
import sounddevice as sd
import time
#from AudioClassifier import AudioClassifier  # Your scream detection class
from fall_detection_system import FallDetectionSystem  # Your fall detection class
from flask import request
import serial

# Discord Webhook for Notifications
'''
WEBHOOK = "https://discord.com/api/webhooks/1329639907442036769/5ShE26g-ZleAN1lY7L5lPGv-HyqZx7TukNTF2rrAwuQeWNUku4dNMrsWZBnHKnJYZOlN"

def get_local_ip():
    """Get local network IP address"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        s.connect(("8.8.8.8", 80))  # Connect to Google's DNS
        ip = s.getsockname()[0]
    except Exception:
        ip = "Unable to determine local IP"
    finally:
        s.close()
    return ip

def discord_message(ip):
    """Send local IP address to Discord webhook"""
    message = {"embeds": [{"title": "IP Address", "color": 65280, "description": f"{ip}:8000"}]}
    x = requests.post(WEBHOOK, json=message)
    print("Discord Notification:", "Success" if x.status_code == 204 else "Failed")
'''

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Video and audio parameters
SAMPLE_RATE = 44100  # Audio sample rate in Hz
CHUNK_SIZE = 1024  # Audio chunk size

# Multiprocessing Queues
video_queue = multiprocessing.Queue(maxsize=10)
audio_queue = multiprocessing.Queue(maxsize=10)

# List to track processes
processes = []

def capture_audio(audio_queue):
    """Capture audio in real-time and store in a queue."""
    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        try:
            audio_queue.put_nowait(indata.tolist())  # Store audio chunks
        except:
            pass  # Avoid blocking if queue is full

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=CHUNK_SIZE):
        while True:
            time.sleep(0.01)  # Prevent CPU overload

def capture_video(video_queue):
    """Capture video frames, process with YOLOv11, and store in a queue."""
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
        
        # Run pose estimation on the captured frame (uses CUDA)
        processed_frame = fall_system.process_frame(frame)

        _, encoded_image = cv2.imencode(".jpg", processed_frame)
        try:
            video_queue.put_nowait(encoded_image.tobytes())  # Store frame in queue
            print("Frame added to queue")  # Debug statement
        except Exception as e:
            print(f"Error adding frame to queue: {e}")
        
        time.sleep(1 / 30)  # Adjust FPS (30 FPS target)

    vc.release()

def emit_data(video_queue, audio_queue):
    """Main thread function to emit audio and video to clients."""
    while True:
        frame = None
        audio_data = None

        #print(f"Video queue size: {video_queue.qsize()}, Audio queue size: {audio_queue.qsize()}")

        if not video_queue.empty():
            frame = video_queue.get_nowait()
            print("Frame exists")

        if not audio_queue.empty():
            audio_data = audio_queue.get_nowait()

        if frame is not None and audio_data is not None:
            try:
                print("Emitting video and audio data...")
                socketio.emit('video_frame', {'frame': frame, 'audio_data': audio_data})
            except Exception as e:
                print(f"Error emitting video and audio data: {e}")
        else:
            pass
        time.sleep(1 / 30)  # Match video FPS


@app.route('/arduino-command', methods=['POST'])
def send_command_to_arduino():
    data = request.get_json()
    command = data.get('command')
    print(command)
    time.sleep(2)
    arduino.write(command.encode())
    print(f"Sent: {command.strip()}")
    time.sleep(1)

    if not command:
        return {"error": "No command provided"}, 400

    try:
        # Send the command to Arduino (ensure it is a string and encoded properly)
        return {"status": "Command sent"}, 200
    except Exception as e:
        return {"error": str(e)}, 500

def graceful_exit(sig, frame):
    """Handles Ctrl + C and stops all processes."""
    print("\n[INFO] Ctrl + C detected. Shutting down...")
    
    for p in processes:
        print(f"[INFO] Terminating process {p.pid}...")
        p.terminate()
        p.join()
    
    print("[INFO] Cleanup complete. Exiting.")
    sys.exit(0)

if __name__ == '__main__':
    # Register Ctrl + C handler
    signal.signal(signal.SIGINT, graceful_exit)

    # Start video and audio processes
    audio_process = multiprocessing.Process(target=capture_audio, args=(audio_queue,))
    video_process = multiprocessing.Process(target=capture_video, args=(video_queue,))
    emit_process = multiprocessing.Process(target=emit_data, args=(video_queue, audio_queue))

    processes.extend([audio_process, video_process, emit_process])  # Track processes

    for p in processes:
        p.start()

    # Run Flask-SocketIO in the main thread
    arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
    socketio.run(app, host="0.0.0.0", port=8000, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)
