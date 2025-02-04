import logging
import torch
from ultralytics import YOLO
import numpy as np

class PoseEstimator:
    """
    Estimates human poses using a YOLO model and returns detection results.
    """

    def __init__(self, model_path: str):
        # Determine the device to run the model on
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Using device: {self.device}")

        # Initialize the YOLO model for pose estimation
        self.model = YOLO(model_path).to(self.device)
        logging.info(f"Model loaded and moved to device: {self.device}")

    def detect_poses(self, frame: np.ndarray):
        """
        Performs pose detection on a given frame.

        Args:
            frame (np.ndarray): The image frame to process.

        Returns:
            tuple: Contains bounding boxes, keypoints, keypoint scores,
                   confidences, and the result object.
        """
        # Perform inference using the YOLO model
        results = self.model.predict(frame, device=self.device)

        if results and results[0].boxes is not None:
            # Extract bounding boxes, keypoints, and related information
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            keypoints = results[0].keypoints.xy.int().cpu().tolist()
            keypoint_scores = results[0].keypoints.conf.cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist()
            return boxes, keypoints, keypoint_scores, confidences, results[0]

        # Return None if no detections are found
        return None, None, None, None, None