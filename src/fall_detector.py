import logging
import numpy as np
import math
from collections import deque

class FallDetector:
    """
    Detects falls based on keypoint positions and body orientation.
    """

    def __init__(self, fall_ratio_threshold=0.6, torso_angle_threshold=70, required_consecutive_frames=1):
        self.fall_ratio_threshold = fall_ratio_threshold
        self.torso_angle_threshold = torso_angle_threshold
        self.required_consecutive_frames = required_consecutive_frames

        # Use deque for efficient history management
        self.fall_frame_counts = {}
        self.torso_angle_history = {}
        self.torso_position_history = {}
        self.history_length = 5

    def update_torso_history(self, track_id: int, angle: float, position_y: float):
        self.torso_angle_history.setdefault(track_id, deque(maxlen=self.history_length)).append(angle)
        self.torso_position_history.setdefault(track_id, deque(maxlen=self.history_length)).append(position_y)

    def is_fallen(self, keypoints: list, keypoint_scores: list, track_id: int) -> bool:
        try:
            # Required keypoints with minimum confidence
            required_keypoints = {
                0: 0.1,   # Nose
                5: 0.1,   # Left Shoulder
                6: 0.1,   # Right Shoulder
                11: 0.1,  # Left Hip
                12: 0.1,  # Right Hip
                15: 0.1,  # Left Ankle
                16: 0.1   # Right Ankle
            }

            # Validate required keypoints
            for idx, min_conf in required_keypoints.items():
                if idx >= len(keypoint_scores) or keypoint_scores[idx] < min_conf:
                    logging.debug(f"Low confidence or missing keypoint {idx}")
                    return False

            # Nose and ankle positions
            nose_y = keypoints[0][1]
            ankle_ys = [keypoints[idx][1] for idx in [15, 16] if keypoint_scores[idx] >= required_keypoints[idx]]
            if not ankle_ys:
                logging.debug("No ankle keypoints available")
                return False
            avg_ankle_y = np.mean(ankle_ys)
            body_height = abs(nose_y - avg_ankle_y)
            fall_threshold = self.fall_ratio_threshold * body_height

            # Check if the nose is near ankle level
            nose_near_ankles = abs(nose_y - avg_ankle_y) < fall_threshold

            # Torso angle calculation
            shoulders = [keypoints[idx] for idx in [5, 6] if keypoint_scores[idx] >= required_keypoints[idx]]
            hips = [keypoints[idx] for idx in [11, 12] if keypoint_scores[idx] >= required_keypoints[idx]]
            if len(shoulders) < 1 or len(hips) < 1:
                logging.debug("Insufficient keypoints for torso angle calculation")
                return False
            shoulder_midpoint = np.mean(shoulders, axis=0)
            hip_midpoint = np.mean(hips, axis=0)

            delta_x = hip_midpoint[0] - shoulder_midpoint[0]
            delta_y = hip_midpoint[1] - shoulder_midpoint[1]
            torso_angle = abs(math.degrees(math.atan2(delta_y, delta_x)))

            # Normalize the angle to be within 0-90 degrees
            torso_angle = torso_angle if torso_angle <= 90 else 180 - torso_angle
            torso_near_horizontal = torso_angle < self.torso_angle_threshold

            # Update torso history
            self.update_torso_history(track_id, torso_angle, hip_midpoint[1])

            if len(self.torso_angle_history[track_id]) < self.history_length:
                return False

            # Changes in torso angle and position
            angle_change = self.torso_angle_history[track_id][-1] - self.torso_angle_history[track_id][0]
            position_change = self.torso_position_history[track_id][-1] - self.torso_position_history[track_id][0]

            # Fall detection criteria
            fall_detected = (nose_near_ankles or torso_near_horizontal) and angle_change > 20 and position_change > 30

            if fall_detected:
                self.fall_frame_counts[track_id] = self.fall_frame_counts.get(track_id, 0) + 1
            else:
                self.fall_frame_counts[track_id] = 0

            return self.fall_frame_counts[track_id] >= self.required_consecutive_frames

        except Exception as e:
            logging.warning(f"Error in fall detection: {e}")
            return False

    def is_sitting(self, keypoints: list, keypoint_scores: list, frame_height: int) -> bool:
        try:
            required_keypoints = {11: 0.1, 12: 0.1}  # Left and Right Hips
            hip_y_positions = [keypoints[idx][1] for idx in required_keypoints if keypoint_scores[idx] >= required_keypoints[idx]]
            if not hip_y_positions:
                return False
            avg_hip_y = np.mean(hip_y_positions)
            sitting_threshold = frame_height * 0.8  # 80% of frame height
            return avg_hip_y > sitting_threshold
        except Exception as e:
            logging.warning(f"Error in sitting detection: {e}")
            return False