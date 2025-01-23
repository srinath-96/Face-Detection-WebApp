# utils/image_processor.py
from ultralytics import YOLO
import numpy as np
import cv2
from typing import Tuple, List
import logging
import torch
import os

class ImageProcessor:
    def __init__(self):
        """Initialize YOLO Face Detection"""
        try:
            # Download YOLOv8-face model if not exists
            if not os.path.exists('yolov8n-face.pt'):
                os.system('wget https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt')
            
            # Load the face detection model
            self.model = YOLO('yolov8n-face.pt')
            logging.info("YOLO face detection model loaded successfully")
            
            # Set device
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logging.info(f"Using device: {self.device}")
            
        except Exception as e:
            logging.error(f"Error initializing YOLO model: {str(e)}")
            raise

    def process_image(self, image: np.ndarray, confidence_threshold: float = 0.5) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Process image to detect faces using YOLOv8-face.
        
        Args:
            image: numpy array of the image (BGR format)
            confidence_threshold: minimum confidence for face detection
            
        Returns:
            Tuple of (processed image, list of face regions)
        """
        try:
            # Make a copy of the image
            img = image.copy()
            height, width = img.shape[:2]
            
            # Run YOLO detection
            results = self.model(img, conf=confidence_threshold)
            
            # Lists to store face regions
            face_regions = []
            
            # Process detections
            for result in results:
                if result.boxes:  # If there are detections
                    boxes = result.boxes
                    for box in boxes:
                        # Get coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        
                        # Ensure coordinates are within image bounds
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(width, x2)
                        y2 = min(height, y2)
                        
                        # Draw bounding box with thicker line
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        
                        # Add confidence score with better visibility
                        label = f"Face: {conf:.2f}"
                        font_scale = 0.8
                        thickness = 2
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        
                        # Get text size
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label, font, font_scale, thickness
                        )
                        
                        # Draw background rectangle for text
                        cv2.rectangle(
                            img,
                            (x1, y1 - text_height - 10),
                            (x1 + text_width, y1),
                            (0, 255, 0),
                            -1
                        )
                        
                        # Add text
                        cv2.putText(
                            img,
                            label,
                            (x1, y1 - 5),
                            font,
                            font_scale,
                            (0, 0, 0),
                            thickness
                        )
                        
                        # Extract face region with margin
                        margin = int((y2 - y1) * 0.2)  # 20% margin
                        face_y1 = max(0, y1 - margin)
                        face_y2 = min(height, y2 + margin)
                        face_x1 = max(0, x1 - margin)
                        face_x2 = min(width, x2 + margin)
                        
                        face_region = img[face_y1:face_y2, face_x1:face_x2].copy()
                        if face_region.size > 0:
                            face_regions.append(face_region)
            
            return img, face_regions
            
        except Exception as e:
            logging.error(f"Error in face detection: {str(e)}")
            return image, []