from edge_impulse_linux.audio import AudioImpulseRunner
import numpy as np
import sounddevice as sd
import queue

MODEL_PATH = "/home/ufset/Desktop/SET_2024-25/src/audio_model.eim"

class AudioClassifier:
    def __init__(self, device_id: int):
        self.device_ID = device_id
        runner = AudioImpulseRunner(MODEL_PATH)
        model_info = runner.init()                
        labels = model_info['model_parameters']['labels']
        print(f"Loaded model: {model_info['project']['owner']} / {model_info['project']['name']}")

    def classify_audio(self, runner):
        model_info = runner.init()
        print(f"Model loaded: {model_info['project']['name']}")
        labels = model_info['model_parameters']['labels']

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

