from edge_impulse_linux.audio import AudioImpulseRunner
import numpy as np
import sounddevice as sd
import queue

class AudioClassifier:
    def __init__(self, device_id: int):
        self.device_ID = device_id

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
    
if __name__ == '__main__':
    print(sd.query_devices())
    device_id = int(input('Device ID: '))
    audioClassifier = AudioClassifier(device_id)
    MODEL_PATH = "/home/ufset/Desktop/SET_2024-25/src/audio_model.eim"
    runner = AudioImpulseRunner(MODEL_PATH)
    audioClassifier.classify_audio
