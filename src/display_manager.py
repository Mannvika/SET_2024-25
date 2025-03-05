import cv2
import cvzone
import numpy as np

class DisplayManager:
    """
    Manages the display of statuses and overlays on the video frames.
    """

    @staticmethod
    def draw_status(frame: np.ndarray, bounding_box: list, statuses: dict, track_id: int, h: float = 0.1):
        """
        Draws the detection status and bounding box on the frame and places the lidar dot relative to the frame dimensions.

        Args:
            frame (np.ndarray): The image frame to draw on.
            bounding_box (list): Bounding box coordinates [x1, y1, x2, y2].
            statuses (dict): Dictionary containing various status indicators (e.g., "fall_status", "injury_status").
            track_id (int): Unique identifier for the tracked person.
            h (float): Vertical offset factor as a fraction of the frame height.
                       For example, h=0.1 means an offset of 10% of the frame height.
        """
        x1, y1, x2, y2 = bounding_box

        # Get actual frame dimensions
        frame_height, frame_width = frame.shape[:2]

        # Compute the x coordinate: the horizontal center of the frame.
        lidar_x = frame_width // 2

        # Compute the y coordinate: the vertical center plus an offset based on the frame height.
        lidar_y = int(frame_height / 2 + frame_height * h)

        # Determine border color based on the status
        if statuses.get('fall_status') == "Fallen" or statuses.get('injury_status') == "Injured":
            box_color = (0, 0, 255)  # Red for alerts
        else:
            box_color = (0, 255, 0)  # Green for normal status

        # Draw the bounding box with the chosen color.
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness=2)

        # Prepare status text.
        status_text = f"Fall: {statuses.get('fall_status', 'Normal')}, Injury: {statuses.get('injury_status', 'Normal')}"

        # Display the track ID and status above the bounding box.
        cvzone.putTextRect(frame, f'ID: {track_id}', (x1, y1 - 45), scale=1, thickness=1)
        cvzone.putTextRect(frame, status_text, (x1, y1 - 25), scale=1, thickness=1)

        # Draw the lidar dot at the computed location.
        cv2.circle(frame, (lidar_x, lidar_y), 5, (57, 255, 20), cv2.FILLED)

        # Optionally, you can uncomment these lines to display guideline lines for visual reference.
        # cv2.line(frame, (lidar_x, 0), (lidar_x, frame_height), (255, 0, 0), 2)
        # cv2.line(frame, (0, lidar_y), (frame_width, lidar_y), (255, 0, 0), 2)
