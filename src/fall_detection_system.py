import logging
import numpy as np
from datetime import datetime
import json
from deep_sort_realtime.deepsort_tracker import DeepSort
from pose_estimator import PoseEstimator
from fall_detector import FallDetector
from injury_analyzer import InjuryAnalyzer
from display_manager import DisplayManager

class FallDetectionSystem:
    """
    Main system class that integrates pose estimation, fall detection, and injury analysis.
    """

    def __init__(self, model_path: str):
        self.pose_estimator = PoseEstimator(model_path)
        self.fall_detector = FallDetector()
        self.injury_analyzer = InjuryAnalyzer()
        self.display_manager = DisplayManager()
        self.detections = []
        self.tracker = DeepSort(max_age=5, n_init=3, max_iou_distance=0.7)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        # Perform pose estimation on the frame
        boxes, keypoints_list, keypoint_scores_list, confidences, result = self.pose_estimator.detect_poses(frame)

        if boxes is None:
            # Clear histories as no persons are detected
            self.fall_detector.torso_angle_history.clear()
            self.fall_detector.torso_position_history.clear()
            self.injury_analyzer.left_ankle_history.clear()
            self.injury_analyzer.right_ankle_history.clear()
            return frame

        frame_height, frame_width = frame.shape[:2]

        # Prepare detections for the tracker
        tracker_detections = []
        for bbox, confidence in zip(boxes, confidences):
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            tracker_bbox = [x1, y1, width, height]  # DeepSort expects [x, y, w, h]
            tracker_detections.append((tracker_bbox, confidence, 'person'))

        # Update tracker
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
            detection_idx = self._find_matching_detection_index(bbox, boxes)
            if detection_idx is not None:
                track_id_to_detection_idx[track_id] = detection_idx

        # Collect statuses and bounding boxes for all detections
        detections_info = []

        for track_id, detection_idx in track_id_to_detection_idx.items():
            bounding_box = boxes[detection_idx]
            keypoints = keypoints_list[detection_idx]
            keypoint_scores = keypoint_scores_list[detection_idx]
            confidence = confidences[detection_idx]

            if len(keypoints) > 16:
                # Perform fall and injury detection
                fall_detected = self.fall_detector.is_fallen(keypoints, keypoint_scores, track_id)
                sitting_detected = self.fall_detector.is_sitting(keypoints, keypoint_scores, frame_height)
                fall_status = "Fallen" if fall_detected or sitting_detected else "Normal"

                injured = fall_detected or sitting_detected or \
                          self.injury_analyzer.detect_injury_color(frame, *bounding_box) or \
                          self.injury_analyzer.is_limping(keypoints, keypoint_scores, track_id) or \
                          self.injury_analyzer.is_curled_up(keypoints, keypoint_scores)

                injury_status = "Injured" if injured else "Normal"

                statuses = {'fall_status': fall_status, 'injury_status': injury_status}

                # Save detection info for later drawing
                detections_info.append({
                    'bounding_box': bounding_box,
                    'statuses': statuses,
                    'track_id': track_id,
                    'confidence': confidence
                })

                # Create detection data for JSON
                detection_data = {
                    'timestamp': datetime.now().isoformat(),
                    'track_id': int(track_id),
                    'bounding_box': bounding_box,
                    'statuses': statuses,
                    'confidence': float(confidence)
                }
                self.detections.append(detection_data)
            else:
                logging.debug("Insufficient keypoints for detailed analysis.")

        # Overlay the skeletons on the frame using the YOLO result
        frame_with_skeletons = result.plot()

        # Now draw the statuses and bounding boxes on the frame_with_skeletons
        for info in detections_info:
            self.display_manager.draw_status(frame_with_skeletons, info['bounding_box'], info['statuses'],
                                             info['track_id'])

        # Save detections to a JSON file
        self._write_detections_to_json()

        return frame_with_skeletons

    def _find_matching_detection_index(self, bbox: list, boxes: list) -> int:
        """
        Finds the index of the detection that matches the given bounding box based on IoU.

        Args:
            bbox (list): Bounding box coordinates [x1, y1, x2, y2].
            boxes (list): List of all bounding boxes in the current frame.

        Returns:
            int: Index of the matching detection, or None if no match is found.
        """
        for idx, box in enumerate(boxes):
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