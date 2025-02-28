from AudioClassifier.py import AudioClassifier
import sys

device_id = 0

def main(argc, argv):
    print(sd.query_devices())
    device_id = int(input("Which device ID would you like to use?"))
    ScreamDetector  = AudioClassifier(device_id)
    ScreamDetector.start_listening()
    
if __name__ == "__main__":
    try:
        main(sys.argv)
    except KeyboardInterrupt:
        print("\nStopped by user.")
