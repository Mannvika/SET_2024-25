import cv2
import socket
import threading
import time
import numpy as np
import sounddevice as sd

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO

from fall_detection_system import FallDetectionSystem

# -----------------------------------------------------------------------------
# Configuration and Global Variables
# -----------------------------------------------------------------------------

# Discord webhook URL used for sending notifications (not actively used in this code sample)
WEBHOOK = "https://discord.com/api/webhooks/1329639907442036769/5ShE26g-ZleAN1lY7L5lPGv-HyqZx7TukNTF2rrAwuQeWNUku4dNMrsWZBnHKnJYZOlN"

# Audio configuration constants: sample rate and chunk size for audio capture
SAMPLE_RATE = 44100  # Audio sample rate in Hz
CHUNK_SIZE = 1024  # Audio chunk size

# Global flags and synchronization objects
saving_data = False  # Flag indicating whether data is currently being saved
video_thread = None  # Thread object for handling the video streaming
video_stop_event = threading.Event()  # Event used to signal the video stream to stop
lock = threading.Lock()  # Lock to synchronize access to shared resources

# -----------------------------------------------------------------------------
# Flask App and SocketIO Setup
# -----------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing to allow requests from other domains
socketio = SocketIO(app, cors_allowed_origins="*")  # Create a SocketIO server for real-time communication


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def get_local_ip():
    """
    Retrieve the local IP address of the host machine by connecting to a public DNS server.

    The function creates a UDP socket and connects to Google's public DNS (8.8.8.8) on port 80.
    Since it is UDP and no data is sent, it only forces the OS to determine the local IP address.

    Returns:
        str: The local IP address if found, or a fallback error message.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.settimeout(0)
        s.connect(("8.8.8.8", 80))  # Connect to Google DNS to determine the outbound IP
        ip = s.getsockname()[0]
    except Exception:
        ip = "Unable to determine local IP"
    finally:
        s.close()  # Always ensure the socket is properly closed
    return ip


def log_message(message):
    """
    Log a message to the console and emit the message to all connected SocketIO clients.

    This function is used throughout the app to both print internal logs on the server
    and send them to any client that is listening for log events.

    Args:
        message (str): The message to log and emit.
    """
    print(message)
    socketio.emit('log', {'message': message})


# -----------------------------------------------------------------------------
# SocketIO Event Handlers
# -----------------------------------------------------------------------------

@socketio.on("connect")
def handle_connect():
    """
    Handle a new SocketIO client connection.

    When a client connects to the server, a log message is sent to the console and
    the connected client receives a log event.
    """
    log_message("Client connected")


@socketio.on("disconnect")
def handle_disconnect():
    """
    Handle a SocketIO client disconnection.

    This function logs the disconnection event so that both the server and clients
    are aware of the change in connection status.
    """
    log_message("Client disconnected")


@socketio.on("controller_input")
def handle_controller_input(data):
    """
    Process controller input received from a client.

    Expects JSON data that includes a "dpad" key with an array of directions pressed.
    Logs the input for further processing or debugging.

    Args:
        data (dict): JSON data sent by the client containing controller input information.
    """
    if "dpad" in data and data["dpad"]:
        dpad_buttons = ", ".join(data["dpad"])
        log_message(f"D-Pad Pressed: {dpad_buttons}")


@socketio.on("movement_command")
# -----------------------------------------------------------------
# SocketIO Event Handlers  –  only the movement handler changed
# -----------------------------------------------------------------
@socketio.on("movement_command")
def handle_movement_command(data):
    """
    data = { "direction": "up" | "down" | "left" | "right",
             "state"     : "move" | "stop" }
    """
    # translate to plain‑English motion words
    direction_map = {
        "up":   "forward",
        "down": "backward",
        "left": "left",
        "right": "right"
    }

    directions = data["direction"]
    state     = data["state"]

    direction = direction_map.get(directions, directions)
    if state == "move":
        log_message(f"moving {direction}")
    else:  # state == "stop"
        log_message(f"stopped {direction}")



@socketio.on("toggle_video")
def handle_toggle_video(data):
    """
    Toggle video streaming based on the action received from the client.

    When the client sends an action ("start" or "stop"), the function either starts
    a new video streaming thread or stops an existing one. It sends the updated video status
    back to the client via the 'video_status' event.

    Args:
        data (dict): Dictionary containing the key "action" with value "start" or "stop".
    """
    global video_thread, video_stop_event
    action = data.get("action")

    if action == "start":
        # If there is no active video thread, start one
        if video_thread is None or not video_thread.is_alive():
            log_message("Starting video stream...")
            video_stop_event.clear()  # Reset the stop event flag
            video_thread = threading.Thread(target=emit_video_frames, daemon=True)
            video_thread.start()
            socketio.emit("video_status", {"status": "running"})
        else:
            log_message("Video stream is already running.")
            socketio.emit("video_status", {"status": "running"})
    elif action == "stop":
        # If the video is already streaming, stop the thread
        if video_thread and video_thread.is_alive():
            log_message("Stopping video stream...")
            video_stop_event.set()  # Signal the video thread to stop
            video_thread.join()  # Wait for the thread to finish
            video_thread = None
            socketio.emit("video_status", {"status": "stopped"})
        else:
            log_message("Video stream is not active.")
            socketio.emit("video_status", {"status": "stopped"})
    else:
        log_message("Invalid video toggle action received.")


def log_inference_time(inference_time):
    """
    Log and emit the inference time (in milliseconds) to both the console and the clients.

    This function is useful for performance monitoring of the video processing pipeline.

    Args:
        inference_time (float): Time taken for an inference in milliseconds.
    """
    message = f"Inference time: {inference_time}ms"
    log_message(message)


# -----------------------------------------------------------------------------
# Audio and Video Capture Functions
# -----------------------------------------------------------------------------

def capture_audio():
    """
    Capture audio in real-time and emit audio data to connected clients.

    Utilizes the sounddevice library to capture audio in chunks defined by CHUNK_SIZE.
    Each chunk is then emitted to the client over SocketIO.

    The function runs continuously until the thread running it is terminated.
    """

    def audio_callback(indata, frames, time_info, status):
        if status:
            print("Audio callback status:", status)
        # Convert the audio numpy array to list (JSON serializable) and emit it.
        socketio.emit('audio_data', indata.tolist())

    # Open the audio input stream using a context manager. The stream remains open as long as the thread is alive.
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=CHUNK_SIZE):
        threading.Event().wait()  # Wait indefinitely, keeping the stream open


def emit_video_frames():
    """
    Capture and process video frames, then stream them to clients via SocketIO.

    Opens the default camera using OpenCV, and for each frame:
      1. Processes the frame with a fall detection system.
      2. Encodes the processed frame as a JPEG image.
      3. Emits the encoded frame to clients.

    The frame rate is approximately 30 FPS, maintained by sleeping for 1/30th of a second between frames.
    The function stops when the global 'video_stop_event' is set.
    """
    model_path = 'yolo11x-pose.pt'
    fall_system = FallDetectionSystem(model_path)
    vc = cv2.VideoCapture(0)  # Open default video capture device

    if not vc.isOpened():
        log_message("Error: Could not open video stream.")
        return

    while not video_stop_event.is_set():
        ret, frame = vc.read()
        if not ret:
            log_message("Warning: Frame capture unsuccessful.")
            break

        # Process the frame through the fall detection system
        processed_frame = fall_system.process_frame(frame)

        with lock:
            # Encode the frame to JPEG format to reduce data size for transmission
            _, encoded_image = cv2.imencode(".jpg", processed_frame)
            # Emit the image frame to the clients using SocketIO
            socketio.emit('video_frame', {'frame': encoded_image.tobytes()})

        time.sleep(1 / 30)  # Pause to achieve ~30 frames per second

    vc.release()  # Release the video capture device
    log_message("Video stream stopped.")


def generate_audio_data():
    """
    Simulate sending real-time audio data to the client.

    This function generates random audio chunks (as a simulation) and emits them
    continuously to mimic a real audio stream. Replace or modify this when integrating
    with an actual audio source.
    """
    while True:
        # Generate a simulated audio chunk with random values
        audio_chunk = np.random.randint(-32768, 32767, CHUNK_SIZE, dtype=np.int16).tobytes()
        socketio.emit('audio_data_to_client', {'audio': audio_chunk})
        time.sleep(0.1)


# -----------------------------------------------------------------------------
# Main Entry Point: Start the Server and Audio Capture Thread
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # Retrieve and log the local network IP address so that clients can connect to the server.
    local_ip = get_local_ip()
    log_message(f"Local Network IP Address: {local_ip}")

    # Start a dedicated thread for capturing audio. This thread runs independently
    # so that audio capture and streaming do not block the main thread.
    audio_thread = threading.Thread(target=capture_audio, daemon=True)
    audio_thread.start()

    # Start the Flask-SocketIO server on all network interfaces (host="0.0.0.0") at port 8000.
    # The server is set to run without using the Flask reloader or debugger for production use.
    socketio.run(app, host="0.0.0.0", port=8000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
