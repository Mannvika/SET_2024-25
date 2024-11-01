from flask import Flask, jsonify, request
from flask_cors import CORS
import random  # Import the random module

app = Flask(__name__)
CORS(app)

# Predefined list of random messages
messages = [
    "Hello, World!",
    "How are you today?",
    "Have a great day!",
    "Keep smiling!",
    "Flask and React make a great pair!",
    "You're doing great!",
    "Believe in yourself!",
    "The sky is the limit!",
]


@app.route('/api/message', methods=['GET'])
def get_message():
    # Choose a random message from the list
    random_message = random.choice(messages)
    return jsonify(message=random_message)


@app.route('/api/message', methods=['POST'])
def update_message():
    global messages
    data = request.json
    new_message = data.get('message')

    # Optionally, you could add the new message to the list
    if new_message:
        messages.append(new_message)

    return jsonify(success=True, messages=messages)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
