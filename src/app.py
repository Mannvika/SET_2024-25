from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO
import cv2
import threading
from fall_detection_system import FallDetectionSystem
from dynamo import save_to_dynamo
import time
import random

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

saving_data = False
lock = threading.Lock()

def log_inference_time(inference_time):
    """Send inference log to frontend via SocketIO"""
    print(f"Sending log: Inference time: {inference_time}ms")  # Print statement before sending log
    socketio.emit("log", {"message": f"Inference time: {inference_time}ms"})
    print(f"Log sent: Inference time: {inference_time}ms")  # Confirm the log is sent

def log_message(message):
    """Helper function to emit logs to frontend"""
    print(message)  # Log to console
    socketio.emit('log', {'message': message}, broadcast=True)  # Send log to React

@app.route('/stream', methods=['GET'])
def stream():
    """Stream video frames to the client."""
    return Response(generate_video(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/save_data', methods=['POST'])
def save_data():
    """Handles data saving process"""
    global saving_data
    command = request.get_json()

    if not command or 'action' not in command:
        log_message("Invalid request received")
        return jsonify({"message": "Invalid request"}), 400

    action = command['action']

    if action == 'start':
        if not saving_data:
            saving_data = True
            log_message("Started saving data...")
            data = {"runID": "A1"}  # Hardcoded runID
            save_to_dynamo(data)  # Send data to DynamoDB
            return jsonify({"message": "Data saving started"}), 200
        else:
            log_message("Data saving is already active.")
            return jsonify({"message": "Data saving is already active"}), 200

    elif action == 'stop':
        if saving_data:
            saving_data = False
            log_message("Stopped saving data.")
            return jsonify({"message": "Data saving stopped"}), 200
        else:
            log_message("No active data saving to stop.")
            return jsonify({"message": "No active data saving to stop"}), 200

    else:
        log_message("Invalid action received.")
        return jsonify({"message": "Invalid action"}), 400

def generate_video():
    """Generate video frames for streaming."""
    model_path = 'yolo11x-pose.pt'
    fall_system = FallDetectionSystem(model_path)
    vc = cv2.VideoCapture(0)

    if not vc.isOpened():
        log_message("Error: Unable to open camera.")
        return

    while True:
        rval, frame = vc.read()
        if not rval:
            log_message("Error: Frame capture failed.")
            break

        # Simulate inference time
        # Sample code for displaying live messages to react frontend
        inference_time = random.uniform(590, 620)  # Example log value
        log_inference_time(inference_time)  # Emit log to frontend


        # Run pose estimation on the captured frame
        processed_frame = fall_system.process_frame(frame)

        with lock:
            _, encoded_image = cv2.imencode(".jpg", processed_frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')

    vc.release()

if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=8000, debug=True, allow_unsafe_werkzeug=True)
