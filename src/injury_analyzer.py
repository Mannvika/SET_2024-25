import logging
import numpy as np
import cv2
from collections import deque

class InjuryAnalyzer:
    """
    Analyzes potential injuries based on color detection and posture analysis.
    """

    def __init__(self, red_intensity_threshold=130, red_difference_threshold=70, history_length=5):
        self.red_intensity_threshold = red_intensity_threshold
        self.red_difference_threshold = red_difference_threshold
        self.left_ankle_history = {}
        self.right_ankle_history = {}
        self.history_length = history_length

    def update_ankle_history(self, track_id: int, left_ankle_y: float, right_ankle_y: float):
        self.left_ankle_history.setdefault(track_id, deque(maxlen=self.history_length)).append(left_ankle_y)
        self.right_ankle_history.setdefault(track_id, deque(maxlen=self.history_length)).append(right_ankle_y)

    def detect_injury_color(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
        injury_detected = False
        try:
            # Define regions for head and torso within the bounding box
            head_region = frame[y1:y1 + 50, x1:x2]
            torso_region = frame[y1 + 50:y1 + 150, x1:x2]

            # Calculate average color in BGR format
            avg_color_head = np.array(cv2.mean(head_region)[:3])
            avg_color_torso = np.array(cv2.mean(torso_region)[:3])

            # Check for high red intensity in the head region
            if (avg_color_head[2] > self.red_intensity_threshold and
                    (avg_color_head[2] - avg_color_head[:2].mean()) > self.red_difference_threshold):
                injury_detected = True

            # Check for high red intensity in the torso region
            if (avg_color_torso[2] > self.red_intensity_threshold and
                    (avg_color_torso[2] - avg_color_torso[:2].mean()) > self.red_difference_threshold):
                injury_detected = True

        except Exception as e:
            logging.warning(f"Error in injury color detection: {e}")

        return injury_detected

    def is_limping(self, keypoints: list, keypoint_scores: list, track_id: int) -> bool:
        try:
            required_keypoints = {15: 0.1, 16: 0.1}  # Left and Right Ankles
            if all(keypoint_scores[idx] >= required_keypoints[idx] for idx in required_keypoints):
                left_ankle_y = keypoints[15][1]
                right_ankle_y = keypoints[16][1]

                # Update ankle history
                self.update_ankle_history(track_id, left_ankle_y, right_ankle_y)

                if len(self.left_ankle_history[track_id]) < self.history_length:
                    return False

                # Calculate standard deviation of ankle movements
                left_std = np.std(np.diff(self.left_ankle_history[track_id]))
                right_std = np.std(np.diff(self.right_ankle_history[track_id]))

                if min(left_std, right_std) == 0:
                    return False

                std_ratio = max(left_std, right_std) / min(left_std, right_std)
                return std_ratio > 1.5

            else:
                logging.debug("Insufficient ankle keypoints for limping detection")
                return False

        except Exception as e:
            logging.warning(f"Error in limping detection: {e}")
            return False

    def is_curled_up(self, keypoints: list, keypoint_scores: list) -> bool:
        try:
            required_keypoints = {
                7: 0.1,   # Left Elbow
                8: 0.1,   # Right Elbow
                13: 0.1,  # Left Knee
                14: 0.1,  # Right Knee
                11: 0.1,  # Left Hip
                12: 0.1   # Right Hip
            }

            hips = [np.array(keypoints[idx]) for idx in [11, 12] if keypoint_scores[idx] >= required_keypoints[idx]]
            if not hips:
                logging.debug("Insufficient hip keypoints for curled-up detection")
                return False
            hip_center = np.mean(hips, axis=0)

            elbows = [np.array(keypoints[idx]) for idx in [7, 8] if keypoint_scores[idx] >= required_keypoints[idx]]
            knees = [np.array(keypoints[idx]) for idx in [13, 14] if keypoint_scores[idx] >= required_keypoints[idx]]

            distance_threshold = 80.0  # Distance threshold to detect curled-up posture

            elbows_near_hips = all(np.linalg.norm(elbow - hip_center) < distance_threshold for elbow in elbows)
            knees_near_hips = all(np.linalg.norm(knee - hip_center) < distance_threshold for knee in knees)

            return elbows_near_hips and knees_near_hips

        except Exception as e:
            logging.warning(f"Error in curled-up detection: {e}")
            return False