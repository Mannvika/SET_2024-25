from edge_impulse_linux.audio import AudioImpulseRunner
import numpy as np
import sounddevice as sd

MODEL_PATH = "audio_model.eim"

def audio_callback(indata, frames, time, status):
    """ Callback function to process live audio input """
    if status:
        print(status)
    audio_data = np.mean(indata, axis=1)  # Convert stereo to mono
    classify_audio(audio_data)

def classify_audio(audio_data):
    """ Runs inference on the audio data """
    with AudioImpulseRunner(MODEL_PATH) as runner:
        model_info = runner.init()
        print(f"Model loaded: {model_info['project']['name']}")

        features = runner.get_features_from_audio(audio_data)
        result = runner.classify(features)

        if "classification" in result["result"]:
            print("Predictions:", result["result"]["classification"])

# Start listening from the microphone
with sd.InputStream(callback=audio_callback, channels=1, samplerate=44100):
    print("Listening... Press Ctrl+C to stop.")
    while True:
        pass
