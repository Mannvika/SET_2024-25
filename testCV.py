import cv2
import cvzone
import numpy as np
import math
import logging
import torch
from ultralytics import YOLO
import json
from datetime import datetime
from deep_sort_realtime.deepsort_tracker import DeepSort

# Configure logging to display information and debug messages
logging.basicConfig(level=logging.INFO)


class PoseEstimator:
    """
    Estimates human poses using a YOLO model and returns detection results.
    """

    def __init__(self, model_path: str):
        """
        Initializes the PoseEstimator with the given YOLO model path.

        Args:
            model_path (str): Path to the YOLO pose estimation model.
        """
        # Determine the device to run the model on (GPU if available, else CPU)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Using device: {self.device}")

        # Initialize the YOLO model for pose estimation
        self.model = YOLO(model_path)
        # Move the model to the selected device
        self.model.to(self.device)
        logging.info(f"Model loaded and moved to device: {self.device}")

    def detect_poses(self, frame: np.ndarray):
        """
        Performs pose detection on a given frame.

        Args:
            frame (np.ndarray): The image frame to process.

        Returns:
            tuple: Contains bounding boxes, keypoints, keypoint scores,
                   class IDs, confidences, and the result object.
        """
        # Perform inference using the YOLO model
        results = self.model.predict(frame, device=self.device)

        if results and results[0].boxes is not None:
            # Extract bounding boxes, keypoints, and related information
            bounding_boxes = results[0].boxes.xyxy.int().cpu().tolist()
            keypoints = results[0].keypoints.xy.int().cpu().tolist()
            keypoint_scores = results[0].keypoints.conf.cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist()
            return bounding_boxes, keypoints, keypoint_scores, class_ids, confidences, results[0]

        # Return None if no detections are found
        return None, None, None, None, None, None


class FallDetector:
    """
    Detects falls based on keypoint positions and body orientation.
    """

    def __init__(self, fall_ratio_threshold=0.6, torso_angle_threshold=70, required_consecutive_frames=1):
        """
        Initializes the FallDetector with specified thresholds.

        Args:
            fall_ratio_threshold (float): Ratio to determine if the nose is near ankle level.
            torso_angle_threshold (float): Angle threshold to determine torso orientation.
            required_consecutive_frames (int): Number of consecutive frames to confirm a fall.
        """
        self.fall_ratio_threshold = fall_ratio_threshold
        self.torso_angle_threshold = torso_angle_threshold
        self.required_consecutive_frames = required_consecutive_frames

        # Dictionaries to keep track of fall-related metrics per track ID
        self.fall_frame_counts = {}
        self.torso_angle_history = {}
        self.torso_position_history = {}
        self.history_length = 5  # Number of frames to keep in history

    def update_torso_history(self, track_id: int, angle: float, position_y: float):
        """
        Updates the history of torso angles and positions for a given track ID.

        Args:
            track_id (int): Unique identifier for the tracked person.
            angle (float): Current torso angle.
            position_y (float): Current Y-position of the torso.
        """
        if track_id not in self.torso_angle_history:
            self.torso_angle_history[track_id] = []
            self.torso_position_history[track_id] = []

        self.torso_angle_history[track_id].append(angle)
        self.torso_position_history[track_id].append(position_y)

        # Maintain history length
        if len(self.torso_angle_history[track_id]) > self.history_length:
            self.torso_angle_history[track_id].pop(0)
            self.torso_position_history[track_id].pop(0)

    def is_fallen(self, keypoints: list, keypoint_scores: list, track_id: int) -> bool:
        """
        Determines if a person has fallen based on keypoints and torso orientation.

        Args:
            keypoints (list): List of keypoint coordinates.
            keypoint_scores (list): List of keypoint confidence scores.
            track_id (int): Unique identifier for the tracked person.

        Returns:
            bool: True if a fall is detected, False otherwise.
        """
        try:
            # Define required keypoints with minimum confidence thresholds
            required_keypoints = {
                0: 0.1,   # Nose
                5: 0.1,   # Left Shoulder
                6: 0.1,   # Right Shoulder
                11: 0.1,  # Left Hip
                12: 0.1,  # Right Hip
                15: 0.1,  # Left Ankle
                16: 0.1   # Right Ankle
            }

            # Validate the presence and confidence of required keypoints
            for idx, min_conf in required_keypoints.items():
                if idx >= len(keypoint_scores) or keypoint_scores[idx] < min_conf:
                    logging.info(f"Low confidence or missing keypoint {idx}")
                    return False

            # Extract Y-coordinate of the nose
            nose_y = keypoints[0][1]

            # Collect available ankle Y-coordinates
            ankle_ys = []
            if keypoint_scores[15] >= required_keypoints[15]:
                ankle_ys.append(keypoints[15][1])
            if keypoint_scores[16] >= required_keypoints[16]:
                ankle_ys.append(keypoints[16][1])

            if not ankle_ys:
                logging.info("No ankle keypoints available")
                return False

            avg_ankle_y = np.mean(ankle_ys)
            body_height = abs(nose_y - avg_ankle_y)
            fall_threshold = self.fall_ratio_threshold * body_height

            # Check if the nose is near ankle level
            nose_near_ankles = abs(nose_y - avg_ankle_y) < fall_threshold

            # Calculate torso angle using shoulder and hip keypoints
            shoulders = []
            if keypoint_scores[5] >= required_keypoints[5]:
                shoulders.append(keypoints[5])
            if keypoint_scores[6] >= required_keypoints[6]:
                shoulders.append(keypoints[6])

            hips = []
            if keypoint_scores[11] >= required_keypoints[11]:
                hips.append(keypoints[11])
            if keypoint_scores[12] >= required_keypoints[12]:
                hips.append(keypoints[12])

            if len(shoulders) < 1 or len(hips) < 1:
                logging.info("Insufficient keypoints for torso angle calculation")
                return False

            # Compute midpoints of shoulders and hips
            shoulder_midpoint = np.mean(shoulders, axis=0)
            hip_midpoint = np.mean(hips, axis=0)

            delta_x = hip_midpoint[0] - shoulder_midpoint[0]
            delta_y = hip_midpoint[1] - shoulder_midpoint[1]
            torso_angle = abs(math.degrees(math.atan2(delta_y, delta_x)))

            # Normalize the angle to be within 0-90 degrees
            if torso_angle > 90:
                torso_angle = 180 - torso_angle

            torso_near_horizontal = torso_angle < self.torso_angle_threshold

            # Logging for debugging purposes
            logging.debug(f"Torso angle: {torso_angle}")
            logging.debug(f"Nose near ankles: {nose_near_ankles}")
            logging.debug(f"Torso near horizontal: {torso_near_horizontal}")

            # Update torso history
            self.update_torso_history(track_id, torso_angle, hip_midpoint[1])

            if len(self.torso_angle_history[track_id]) < self.history_length:
                # Not enough historical data to confirm a fall
                return False

            # Calculate changes in torso angle and position over the history
            angle_change = self.torso_angle_history[track_id][-1] - self.torso_angle_history[track_id][0]
            position_change = self.torso_position_history[track_id][-1] - self.torso_position_history[track_id][0]

            # Determine if a fall is detected based on criteria
            fall_detected = (nose_near_ankles or torso_near_horizontal) and angle_change > 20 and position_change > 30

            if fall_detected:
                self.fall_frame_counts[track_id] = self.fall_frame_counts.get(track_id, 0) + 1
            else:
                self.fall_frame_counts[track_id] = 0

            # Confirm fall if the required number of consecutive frames meet the criteria
            return self.fall_frame_counts[track_id] >= self.required_consecutive_frames

        except Exception as e:
            logging.warning(f"Error in fall detection: {e}")
            return False

    def is_sitting(self, keypoints: list, keypoint_scores: list, frame_height: int) -> bool:
        """
        Determines if a person is sitting based on hip keypoints.

        Args:
            keypoints (list): List of keypoint coordinates.
            keypoint_scores (list): List of keypoint confidence scores.
            frame_height (int): Height of the video frame.

        Returns:
            bool: True if sitting is detected, False otherwise.
        """
        try:
            # Define required keypoints for hips with minimum confidence
            required_keypoints = {11: 0.1, 12: 0.1}  # Left and Right Hips
            hip_y_positions = []
            for idx, min_conf in required_keypoints.items():
                if idx < len(keypoint_scores) and keypoint_scores[idx] >= min_conf:
                    hip_y_positions.append(keypoints[idx][1])
                else:
                    logging.info(f"Low confidence or missing keypoint {idx}")

            if not hip_y_positions:
                return False

            avg_hip_y = np.mean(hip_y_positions)

            # Define threshold to determine if hips are near the bottom of the frame
            sitting_threshold = frame_height * 0.8  # 80% of frame height

            sitting_detected = avg_hip_y > sitting_threshold

            logging.debug(f"Sitting detected: {sitting_detected}")
            return sitting_detected

        except Exception as e:
            logging.warning(f"Error in sitting detection: {e}")
            return False


class InjuryAnalyzer:
    """
    Analyzes potential injuries based on color detection and posture analysis.
    """

    def __init__(self,
                 red_intensity_threshold: int = 130,
                 red_difference_threshold: int = 70,
                 curl_up_distance_threshold: float = 80.0,
                 limping_ankle_diff_threshold: float = 20.0):
        """
        Initializes the InjuryAnalyzer with specified thresholds.

        Args:
            red_intensity_threshold (int): Threshold for red color intensity.
            red_difference_threshold (int): Threshold for red color difference.
            curl_up_distance_threshold (float): Distance threshold to detect curled-up posture.
            limping_ankle_diff_threshold (float): Difference threshold to detect limping.
        """
        self.red_intensity_threshold = red_intensity_threshold
        self.red_difference_threshold = red_difference_threshold
        self.curl_up_distance_threshold = curl_up_distance_threshold
        self.limping_ankle_diff_threshold = limping_ankle_diff_threshold

        # Dictionaries to keep track of ankle positions per track ID
        self.left_ankle_history = {}
        self.right_ankle_history = {}
        self.history_length = 5  # Number of frames to keep in history

    def update_ankle_history(self, track_id: int, left_ankle_y: float, right_ankle_y: float):
        """
        Updates the history of ankle Y-positions for a given track ID.

        Args:
            track_id (int): Unique identifier for the tracked person.
            left_ankle_y (float): Current Y-position of the left ankle.
            right_ankle_y (float): Current Y-position of the right ankle.
        """
        if track_id not in self.left_ankle_history:
            self.left_ankle_history[track_id] = []
            self.right_ankle_history[track_id] = []

        self.left_ankle_history[track_id].append(left_ankle_y)
        self.right_ankle_history[track_id].append(right_ankle_y)

        # Maintain history length
        if len(self.left_ankle_history[track_id]) > self.history_length:
            self.left_ankle_history[track_id].pop(0)
            self.right_ankle_history[track_id].pop(0)

    def detect_injury_color(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
        """
        Detects potential injuries based on red color intensity in specific regions.

        Args:
            frame (np.ndarray): The image frame being analyzed.
            x1 (int): Top-left X-coordinate of the bounding box.
            y1 (int): Top-left Y-coordinate of the bounding box.
            x2 (int): Bottom-right X-coordinate of the bounding box.
            y2 (int): Bottom-right Y-coordinate of the bounding box.

        Returns:
            bool: True if potential injury is detected, False otherwise.
        """
        injury_detected = False

        try:
            # Define regions for head and torso within the bounding box
            head_region = frame[y1:y1 + 50, x1:x2]
            torso_region = frame[y1 + 50:y1 + 150, x1:x2]

            # Calculate average color in BGR format
            avg_color_head = cv2.mean(head_region)[:3]
            avg_color_torso = cv2.mean(torso_region)[:3]

            # Check for high red intensity in the head region
            if (avg_color_head[2] > self.red_intensity_threshold and
                    (avg_color_head[2] - np.mean([avg_color_head[0], avg_color_head[1]])) > self.red_difference_threshold):
                injury_detected = True
                cv2.putText(frame, "Potential Head Injury", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Check for high red intensity in the torso region
            if (avg_color_torso[2] > self.red_intensity_threshold and
                    (avg_color_torso[2] - np.mean([avg_color_torso[0], avg_color_torso[1]])) > self.red_difference_threshold):
                injury_detected = True
                cv2.putText(frame, "Potential Torso Injury", (x1, y1 + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        except Exception as e:
            logging.warning(f"Error in injury color detection: {e}")

        return injury_detected

    def is_limping(self, keypoints: list, keypoint_scores: list, track_id: int) -> bool:
        """
        Determines if a person is limping based on ankle position variations.

        Args:
            keypoints (list): List of keypoint coordinates.
            keypoint_scores (list): List of keypoint confidence scores.
            track_id (int): Unique identifier for the tracked person.

        Returns:
            bool: True if limping is detected, False otherwise.
        """
        try:
            # Define required keypoints for ankles with minimum confidence
            required_keypoints = {15: 0.1, 16: 0.1}  # Left and Right Ankles
            available_ankles = []

            for idx, min_conf in required_keypoints.items():
                if idx < len(keypoint_scores) and keypoint_scores[idx] >= min_conf:
                    available_ankles.append(keypoints[idx][1])
                else:
                    logging.info(f"Low confidence or missing keypoint {idx}")

            if len(available_ankles) < 2:
                logging.info("Insufficient ankle keypoints for limping detection")
                return False

            # Extract Y-positions of left and right ankles
            left_ankle_y = keypoints[15][1]
            right_ankle_y = keypoints[16][1]

            # Update ankle history
            self.update_ankle_history(track_id, left_ankle_y, right_ankle_y)

            if len(self.left_ankle_history[track_id]) < self.history_length:
                # Not enough historical data to confirm limping
                return False

            # Calculate differences in ankle positions over the history
            left_diffs = np.diff(self.left_ankle_history[track_id])
            right_diffs = np.diff(self.right_ankle_history[track_id])

            # Calculate standard deviation of the differences
            left_std = np.std(left_diffs)
            right_std = np.std(right_diffs)

            # Avoid division by zero by adding a small epsilon
            std_ratio = max(left_std, right_std) / (min(left_std, right_std) + 1e-5)

            # Determine if limping is detected based on standard deviation ratio
            limping_detected = std_ratio > 1.5

            logging.debug(f"Limping detected: {limping_detected}")
            return limping_detected

        except Exception as e:
            logging.warning(f"Error in limping detection: {e}")
            return False

    def is_curled_up(self, keypoints: list, keypoint_scores: list) -> bool:
        """
        Determines if a person is in a curled-up position based on keypoint proximity.

        Args:
            keypoints (list): List of keypoint coordinates.
            keypoint_scores (list): List of keypoint confidence scores.

        Returns:
            bool: True if the person is curled up, False otherwise.
        """
        try:
            # Define required keypoints with minimum confidence thresholds
            required_keypoints = {
                7: 0.1,   # Left Elbow
                8: 0.1,   # Right Elbow
                13: 0.1,  # Left Knee
                14: 0.1,  # Right Knee
                11: 0.1,  # Left Hip
                12: 0.1   # Right Hip
            }

            available_elbows = []
            available_knees = []
            hips = []

            for idx, min_conf in required_keypoints.items():
                if idx < len(keypoint_scores) and keypoint_scores[idx] >= min_conf:
                    if idx in [7, 8]:
                        available_elbows.append(np.array(keypoints[idx]))
                    elif idx in [13, 14]:
                        available_knees.append(np.array(keypoints[idx]))
                    elif idx in [11, 12]:
                        hips.append(np.array(keypoints[idx]))
                else:
                    logging.info(f"Low confidence or missing keypoint {idx}")

            if len(hips) < 1:
                logging.info("Insufficient hip keypoints for curled-up detection")
                return False

            # Calculate the center point of the hips
            hip_center = np.mean(hips, axis=0)

            # Check if elbows and knees are near the hip center within the distance threshold
            elbows_near_hips = all(
                np.linalg.norm(elbow - hip_center) < self.curl_up_distance_threshold
                for elbow in available_elbows
            )

            knees_near_hips = all(
                np.linalg.norm(knee - hip_center) < self.curl_up_distance_threshold
                for knee in available_knees
            )

            curled_up_detected = elbows_near_hips and knees_near_hips

            logging.debug(f"Curled up detected: {curled_up_detected}")
            return curled_up_detected

        except Exception as e:
            logging.warning(f"Error in curled-up detection: {e}")
            return False


class DisplayManager:
    """
    Manages the display of statuses and overlays on the video frames.
    """

    @staticmethod
    def draw_status(frame: np.ndarray, bounding_box: list, fall_status: str, track_id: int, injury_status: str):
        """
        Draws the detection status and bounding box on the frame.

        Args:
            frame (np.ndarray): The image frame to draw on.
            bounding_box (list): Bounding box coordinates [x1, y1, x2, y2].
            fall_status (str): Status indicating if the person has fallen.
            track_id (int): Unique identifier for the tracked person.
            injury_status (str): Status indicating if the person is injured.
        """
        x1, y1, x2, y2 = bounding_box
        box_color = (0, 255, 0)  # Green for normal status

        # Change box color to red if a fall or injury is detected
        if fall_status == "Fallen" or injury_status == "Injured":
            box_color = (0, 0, 255)  # Red for alert

        # Draw the bounding box around the person
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # Prepare status text
        status_text = f"{fall_status}, {injury_status}"

        # Display the track ID above the bounding box
        cvzone.putTextRect(frame, f'ID: {track_id}', (x1, y1 - 45), scale=1, thickness=1)

        # Display the status below the track ID
        cvzone.putTextRect(frame, status_text, (x1, y1 - 25), scale=1, thickness=1)


class FallDetectionSystem:
    """
    Main system class that integrates pose estimation, fall detection, and injury analysis.
    """

    def __init__(self, model_path: str):
        """
        Initializes the FallDetectionSystem with the specified YOLO model path.

        Args:
            model_path (str): Path to the YOLO pose estimation model.
        """
        self.pose_estimator = PoseEstimator(model_path)
        self.fall_detector = FallDetector()
        self.injury_analyzer = InjuryAnalyzer()
        self.display_manager = DisplayManager()
        self.detections = []  # List to store detection data
        # Initialize DeepSort tracker with specified parameters
        self.tracker = DeepSort(max_age=5, n_init=3, max_iou_distance=0.7)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Processes a single frame for fall and injury detection.

        Args:
            frame (np.ndarray): The image frame to process.

        Returns:
            np.ndarray: The processed frame with overlays.
        """
        # Perform pose estimation on the frame
        bounding_boxes, keypoints_list, keypoint_scores_list, class_ids, confidences, result = self.pose_estimator.detect_poses(frame)

        if bounding_boxes is None:
            logging.info("No detections available.")
            # Clear histories as no persons are detected
            self.fall_detector.torso_angle_history.clear()
            self.fall_detector.torso_position_history.clear()
            self.injury_analyzer.left_ankle_history.clear()
            self.injury_analyzer.right_ankle_history.clear()
            return frame

        frame_height, frame_width = frame.shape[:2]

        # Prepare detections for the tracker in the format (bbox, confidence, class)
        tracker_detections = []
        for bbox, confidence in zip(bounding_boxes, confidences):
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            tracker_bbox = [x1, y1, width, height]  # DeepSort expects [x, y, w, h]
            tracker_detections.append((tracker_bbox, confidence, 'person'))

        # Update tracker with current frame detections
        tracks = self.tracker.update_tracks(tracker_detections, frame=frame)

        # Map track IDs to their corresponding detection indices
        track_id_to_detection_idx = {}
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            bbox = [x1, y1, x2, y2]
            # Find the detection index that matches the current bounding box
            detection_idx = self._find_matching_detection_index(bbox, bounding_boxes)
            if detection_idx is not None:
                track_id_to_detection_idx[track_id] = detection_idx

        # Iterate through each tracked detection
        for track_id, detection_idx in track_id_to_detection_idx.items():
            bounding_box = bounding_boxes[detection_idx]
            keypoints = keypoints_list[detection_idx]
            keypoint_scores = keypoint_scores_list[detection_idx]
            confidence = confidences[detection_idx]

            if len(keypoints) > 16:
                # Perform fall detection
                fall_detected = self.fall_detector.is_fallen(keypoints, keypoint_scores, track_id)
                sitting_detected = self.fall_detector.is_sitting(keypoints, keypoint_scores, frame_height)

                # Determine fall status based on detections
                if fall_detected or sitting_detected:
                    fall_status = "Fallen"
                else:
                    fall_status = "Normal"

                # Initialize injury status
                injured = False

                # Mark as injured if a fall is detected
                if fall_detected:
                    injured = True

                # Detect potential injuries based on color analysis
                if self.injury_analyzer.detect_injury_color(frame, *bounding_box):
                    injured = True

                # Check for limping
                if self.injury_analyzer.is_limping(keypoints, keypoint_scores, track_id):
                    injured = True

                # Check if the person is curled up
                if self.injury_analyzer.is_curled_up(keypoints, keypoint_scores):
                    injured = True

                # Set injury status based on detections
                injury_status = "Injured" if injured else "Normal"

                # Create a dictionary to store detection data
                detection_data = {
                    'timestamp': datetime.now().isoformat(),
                    'track_id': int(track_id),
                    'bounding_box': bounding_box,
                    'fall_status': fall_status,
                    'injury_status': injury_status,
                    'confidence': float(confidence)
                }

                # Append the detection data to the list
                self.detections.append(detection_data)

                # Draw the status and bounding box on the frame
                self.display_manager.draw_status(frame, bounding_box, fall_status, track_id, injury_status)
            else:
                logging.info("Insufficient keypoints for detailed analysis.")

        # Overlay the skeletons on the frame using the YOLO result
        frame_with_skeletons = result.plot()

        # Save detections to a JSON file
        self._write_detections_to_json()

        return frame_with_skeletons

    def _find_matching_detection_index(self, bbox: list, bounding_boxes: list) -> int:
        """
        Finds the index of the detection that matches the given bounding box based on IoU.

        Args:
            bbox (list): Bounding box coordinates [x1, y1, x2, y2].
            bounding_boxes (list): List of all bounding boxes in the current frame.

        Returns:
            int: Index of the matching detection, or None if no match is found.
        """
        for idx, box in enumerate(bounding_boxes):
            iou = self._compute_iou(bbox, box)
            if iou > 0.5:
                return idx
        return None

    @staticmethod
    def _compute_iou(box_a: list, box_b: list) -> float:
        """
        Computes the Intersection over Union (IoU) between two bounding boxes.

        Args:
            box_a (list): First bounding box [x1, y1, x2, y2].
            box_b (list): Second bounding box [x1, y1, x2, y2].

        Returns:
            float: IoU value.
        """
        xA = max(box_a[0], box_b[0])
        yA = max(box_a[1], box_b[1])
        xB = min(box_a[2], box_b[2])
        yB = min(box_a[3], box_b[3])

        # Compute the area of intersection
        inter_width = max(0, xB - xA)
        inter_height = max(0, yB - yA)
        inter_area = inter_width * inter_height

        if inter_area == 0:
            return 0.0

        # Compute the area of both bounding boxes
        box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        # Compute the IoU
        iou = inter_area / float(box_a_area + box_b_area - inter_area)
        return iou

    def _write_detections_to_json(self):
        """
        Writes the collected detection data to a JSON file.
        """
        json_file_path = 'detections.json'

        # Attempt to read existing data from the JSON file
        try:
            with open(json_file_path, 'r') as json_file:
                existing_data = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = {}

        # Update existing data with new detections
        for detection in self.detections:
            track_id = str(detection['track_id'])
            if track_id not in existing_data:
                existing_data[track_id] = []
            existing_data[track_id].append(detection)

        # Write the updated data back to the JSON file
        try:
            with open(json_file_path, 'w') as json_file:
                json.dump(existing_data, json_file, indent=4)
        except Exception as e:
            logging.error(f"Error writing detections to JSON: {e}")

        # Clear the detections list after writing to the file
        self.detections.clear()


def main():
    """
    Main function to run the Fall Detection System.
    """
    # Path to the YOLO pose estimation model
    model_path = '../yolo11x-pose (1).pt'

    # Initialize the Fall Detection System
    fall_detection_system = FallDetectionSystem(model_path)

    # Initialize video capture from the default webcam
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        logging.error("Error: Could not open webcam.")
        return

    logging.info("Fall Detection System is running. Press 'q' to quit.")

    while True:
        # Read a frame from the webcam
        ret, frame = video_capture.read()
        if not ret:
            logging.error("Error: Could not read frame from webcam.")
            break

        # Process the frame for fall and injury detection
        processed_frame = fall_detection_system.process_frame(frame)

        # Display the processed frame in a window
        cv2.imshow('Fall Detection System', processed_frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Exiting Fall Detection System.")
            break

    # Release the video capture object and close all OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



