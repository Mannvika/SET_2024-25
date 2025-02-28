'''
if sunny deletes this he will be hardlocked in gold 3 4ever
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
            score = res['result']['classification'][label]
            print(score)
            for label in labels:
                if label == 'notScreaming':
                    continue
                score = res['result']['classification'][label]
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
