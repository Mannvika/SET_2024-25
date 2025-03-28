import serial
import time

class LidarRead:
    """
    Reads LiDar distances from Arduino
    """
    def __init__(self, _port: str, _baudrate: int, _timeout: float):
        self.arduino = serial.Serial(port=_port, baudrate=_baudrate, timeout=_timeout)
        time.sleep(0.5) #reset time 

    def lidar_read(self):
        try:
            while True:
                data = self.arduino.readline().decode('utf-8').strip()
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
        except KeyboardInterrupt:
            print("\nStopping LiDAR read process.")
        finally:
            self.arduino.close()
            print("Serial connection closed.")
