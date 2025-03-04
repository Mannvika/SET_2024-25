from edge_impulse_linux.audio import AudioImpulseRunner
import numpy as np
import sounddevice as sd
import queue

MODEL_PATH = "/home/ufset/Desktop/SET_2024-25/src/audio_model.eim"

class AudioClassifier:
    def __init__(self, device_id: int):
        self.device_ID = device_id
        self.audio_queue = queue.Queue()

    def audio_callback(self, indata, frames, time, status):
        if status:
            print("Audio Input Error:", status)
        self.audio_queue.put(indata.copy())

    def classify_audio(self, runner):
        model_info = runner.init()
        print(f"Model loaded: {model_info['project']['name']}")
        labels = model_info['model_parameters']['labels']
        
        if self.audio_queue.empty():
            return []
        
        audio_data = self.audio_queue.get()
        audio_data = np.mean(audio_data, axis=1)

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

    def start_listening(self, audio_queue):
        with AudioImpulseRunner(MODEL_PATH) as runner:
            with sd.InputStream(callback=self.audio_callback, channels=2, samplerate=44100, blocksize=512, device='default'):
                print("Listening...")
                while True:
                    scores = self.classify_audio(runner)
                    for score in scores:
                        if score >= 0.7:
                            print("This is actually a scream post-processing.")
                            audio_queue.put("Scream detected!")


def audio_process(audio_queue):
    """Runs scream detection in a separate process"""
    device_id = 0  # Adjust as needed
    scream_detector = AudioClassifier(device_id)
    scream_detector.start_listening(audio_queue)
