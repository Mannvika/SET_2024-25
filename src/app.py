from edge_impulse_linux.audio import AudioImpulseRunner
import numpy as np
import sounddevice as sd
import queue

MODEL_PATH = "/app/audio_model.eim"

# Queue for storing audio chunks
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """ Callback function to process live audio input """
    if status:
        print("Audio Input Error:", status)
    audio_queue.put(indata.copy())  # Store audio data in the queue

def classify_audio(runner):
    """ Continuously processes the audio queue and runs inference """
    model_info = runner.init()
    print(f"Model loaded: {model_info['project']['name']}")

    while True:
        audio_data = audio_queue.get()  # Retrieve audio data from the queue
        audio_data = np.mean(audio_data, axis=1)  # Convert stereo to mono

        features = runner.get_features_from_audio(audio_data)
        result = runner.classify(features)

        if "classification" in result["result"]:
            print("Predictions:", result["result"]["classification"])

def main():
    with AudioImpulseRunner(MODEL_PATH) as runner:
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=16000, blocksize=512):
            print("Listening... Press Ctrl+C to stop.")
            classify_audio(runner)  # Start classification loop

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
