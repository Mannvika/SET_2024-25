from edge_impulse_linux.audio import AudioImpulseRunner
import numpy as np
import sounddevice as sd
import queue

class AudioClassifier:
    def __init__(self, device_id: int):
        self.device_ID = device_id
        self.audio_queue = queue.Queue()

    def classify_audio(self, runner):
        model_info = runner.init()
        print(f"Model loaded: {model_info['project']['name']}")
        labels = model_info['model_parameters']['labels']
        
        if self.audio_queue.empty():
            print("Queue empty")
            return []
        
        audio_data = self.audio_queue.get()
        audio_data = np.mean(audio_data, axis=1)

        screaming_scores = []
        for res, audio in runner.classifier(self.device_ID):
            print('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='\n')
            print("Audio Type " + type(audio) + " Audio Info: " + audio)
            for label in labels:
                score = res['result']['classification'][label]
                print(score, label)
                if label == 'notScreaming':
                    continue
                screaming_scores.append(score)

        print("completed classify audio")
        return screaming_scores
