import cv2
import cvzone
import numpy as np

class DisplayManager:
    """
    Manages the display of statuses and overlays on the video frames.
    """

    @staticmethod
    def draw_status(frame: np.ndarray, bounding_box: list, statuses: dict, track_id: int):
        """
        Draws the detection status and bounding box on the frame.

        Args:
            frame (np.ndarray): The image frame to draw on.
            bounding_box (list): Bounding box coordinates [x1, y1, x2, y2].
            statuses (dict): Dictionary containing various status indicators (e.g., "fall_status", "injury_status").
            track_id (int): Unique identifier for the tracked person.
        """
        x1, y1, x2, y2 = bounding_box

        # Determine border color based on the status
        if statuses.get('fall_status') == "Fallen" or statuses.get('injury_status') == "Injured":
            box_color = (0, 0, 255)  # Red for alerts
        else:
            box_color = (0, 255, 0)  # Green for normal status

        # Draw the bounding box with the chosen color
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness=2)

        # Prepare status text
        status_text = f"Fall: {statuses.get('fall_status', 'Normal')}, Injury: {statuses.get('injury_status', 'Normal')}"

        # Display the track ID and status above the bounding box
        cvzone.putTextRect(frame, f'ID: {track_id}', (x1, y1 - 45), scale=1, thickness=1)
        cvzone.putTextRect(frame, status_text, (x1, y1 - 25), scale=1, thickness=1)