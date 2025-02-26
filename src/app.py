from edge_impulse_linux.audio import AudioImpulseRunner
import numpy as np
import sounddevice as sd
import queue

MODEL_PATH = "/home/ufset/Desktop/SET_2024-25/src/audio_model.eim"

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

    labels = model_info['model_parameters']['labels']
    
    while True:
        audio_data = audio_queue.get()  # Retrieve audio data from the queue
        audio_data = np.mean(audio_data, axis=1)  # Convert stereo to mono

        for res, audio in runner.classifier(0):
            print('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='\n')
            for label in labels:
               score = res['result']['classification'][label]
               print('%s: %.2f\t' % (label, score), end='\n')

def main():
    with AudioImpulseRunner(MODEL_PATH) as runner:
        with sd.InputStream(callback=audio_callback, channels=2, samplerate=44100, blocksize=512, device='default'):
            print("Listening... Press Ctrl+C to stop.")
            classify_audio(runner)  # Start classification loop

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
