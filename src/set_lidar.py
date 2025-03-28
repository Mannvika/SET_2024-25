import serial
import time

def main():
    arduino = serial.Serial(port='COM5', baudrate=9600, timeout=0.1)
    time.sleep(0.5)  # reset time 

    while True:
        data = arduino.readline().decode('utf-8').strip()
        if data:
            try:
                parts = data.split('\t')

                distance_str = parts[0].split('=')[1].strip()
                strength_str = parts[1].split('=')[1].strip()

                distance = int(distance_str)
                strength = int(strength_str)

                print(f"Distance: {distance} cm, Strength: {strength}")
            except (IndexError, ValueError) as e:
                print(f"Error parsing data: {e}")
        time.sleep(0.05)

if __name__ == "__main__":
    main()