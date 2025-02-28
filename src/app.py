from AudioClassifier.py import AudioClassifier
import sys

device_id = 0

def main(argc, argv[]):
    device_id = argv[1]
    ScreamDetector  = AudioClassifier(device_id)
    ScreamDetector.start_listening()
    
if __name__ == "__main__":
    try:
        main(sys.argv)
    except KeyboardInterrupt:
        print("\nStopped by user.")
