from AudioClassifier import AudioClassifier
import sys

def main():
    print(sd.query_devices())
    device_id = int(input("Which device ID would you like to use?"))
    ScreamDetector  = AudioClassifier(device_id)
    ScreamDetector.start_listening()
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
