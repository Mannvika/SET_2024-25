import pyaudio
import wave

def list_audio_devices():
    """List the available audio input devices."""
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    print("Available audio input devices:")
    for i in range(0, num_devices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        if device_info['maxInputChannels'] > 0:
            print(f"ID: {i}, Name: {device_info['name']}")
    p.terminate()

def record_audio(device_id, output_filename="output.wav", record_seconds=5, channels=1, rate=44100, frames_per_buffer=4096):
    """Record audio from the specified device ID."""
    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=rate,
                    input=True,
                    input_device_index=device_id,
                    frames_per_buffer=frames_per_buffer)

    print("Recording...")
    frames = []

    for i in range(0, int(rate / frames_per_buffer * record_seconds)):
        try:
            data = stream.read(frames_per_buffer)
            frames.append(data)
        except IOError as e:
            print(f"Error recording: {e}")

    print("Finished recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

if __name__ == "__main__":
    list_audio_devices()
    device_id = int(input("Enter the device ID you want to use for recording: "))
    record_audio(device_id)
    print("Audio recorded and saved as output.wav")