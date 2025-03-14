'''
'''
'''
if sunny deletes this he will be hardlocked in gold 3 4ever
'''
'''
from edge_impulse_linux.audio import AudioImpulseRunner
import numpy as np
import sounddevice as sd
import queue

MODEL_PATH = "/home/ufset/Desktop/SET_2024-25/src/audio_model.eim"


class AudioClassifier:

    def __init__(self, device_id: int):
        self.device_ID = device_id
        # Queue for storing audio chunks
        self.audio_queue = queue.Queue()
        return

    def audio_callback(self, indata, frames, time, status):
        # This method is used to take audio input
        if status:
            print("Audio Input Error:", status)
        self.audio_queue.put(indata.copy())  # Store audio data in the queue

    def classify_audio(self, runner):
        """ Continuously processes the audio queue and runs inference """
        model_info = runner.init()
        print(f"Model loaded: {model_info['project']['name']}")
        labels = model_info['model_parameters']['labels']
        audio_data = self.audio_queue.get()  # Retrieve audio data from the queue
        audio_data = np.mean(audio_data, axis=1)  # Convert stereo to mono

        screaming_scores = []
        for res, audio in runner.classifier(self.device_ID):
            print('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='\n')
            for label in labels:
                score = res['result']['classification'][label]
                print(score, label)
                if label == 'notScreaming':
                    continue
                screaming_scores.append(score)
        return screaming_scores

    def start_listening(self):
        # Establish Model
        with AudioImpulseRunner(MODEL_PATH) as runner:
            # Create Microphone Input Stream and start streaming
            with sd.InputStream(callback=self.audio_callback, channels=2, samplerate=44100, blocksize=512, device='default'):
                print("Listening...")
                # Call classify_audio to start...y'know, classifying audio.
                while True:
                    print("Doing a classification function.")
                    scores = self.classify_audio(runner)
                    for score in scores:
                        if score >= 0.7:
                            print("This is actually a scream post-processing.")
                    if KeyboardInterrupt:
                        # keeping this just for testing purposes
                        print("\nStopped by user.")
'''

import queue
import numpy as np
import sounddevice as sd
from edge_impulse_linux.audio import AudioImpulseRunner

MODEL_PATH = "/home/ufset/Desktop/SET_2024-25/src/audio_model.eim"

class AudioClassifier:
    def __init__(self, device_id: int, sample_rate=44100, duration=1.0):
        self.device_ID = device_id
        self.sample_rate = sample_rate
        self.duration = duration  # Duration of recording in seconds
        self.audio_queue = queue.Queue()
        self.temp_audio = []  # Temporary storage for audio before sending to queue
        self.frames_needed = int(sample_rate * duration)  # Number of frames needed
        self.recording_complete = False  # Flag to stop after gathering enough data

    def audio_callback(self, indata, frames, time, status):
        """Capture audio and store it until 1 second of data is collected."""
        if status:
            print("Audio Input Error:", status)

        self.temp_audio.append(indata.copy())  # Store incoming data

        # Flatten list and check if we have enough frames
        total_audio = np.concatenate(self.temp_audio, axis=0)
        if len(total_audio) >= self.frames_needed:
            self.audio_queue.put(total_audio[:self.frames_needed])  # Store exactly 1 second
            self.recording_complete = True  # Stop recording
            print("1 second of audio collected.")

    def record_audio_once(self):
        """Start the microphone, gather exactly 1 second of audio, then stop."""
        self.temp_audio = []  # Clear previous audio
        self.recording_complete = False  # Reset flag

        print("Recording audio for classification...")
        with sd.InputStream(callback=self.audio_callback, channels=2, samplerate=self.sample_rate, blocksize=512):
            while not self.recording_complete:
                sd.sleep(100)  # Let audio input run until 1 second is collected

    def classify_audio(self, runner):
        """Process a single audio sample and return one classification."""
        if self.audio_queue.empty():
            print("No audio data available. Recording new sample...")
            self.record_audio_once()  # Record if no audio is in the queue

        if self.audio_queue.empty():  # Double-check in case of failure
            print("Failed to collect audio.")
            return None

        model_info = runner.init()
        print(f"Model loaded: {model_info['project']['name']}")
        labels = model_info['model_parameters']['labels']

        audio_data = self.audio_queue.get()  # Retrieve recorded audio
        audio_data = np.mean(audio_data, axis=1)  # Convert stereo to mono

        # Run inference
        res = runner.classifier(audio_data)
        if res is None:
            print("Inference failed.")
            return None
            
        for res, audio in runner.classifier(self.device_ID):
            print(f'Result ({res["timing"]["dsp"] + res["timing"]["classification"]} ms)')

            # Get classification with highest probability
            classification = max(res['result']['classification'], key=res['result']['classification'].get)
            score = res['result']['classification'][classification]

        print(f"Detected: {classification} ({score:.2f})")
        return classification, score
