from ultralytics import YOLO
import numpy as np
import cv2
from typing import Tuple, List
import logging
import torch
import os
from utils.face_analyzer import FaceAnalyzer  # Changed from relative to absolute import

class ImageProcessor:
    def __init__(self):
        """Initialize YOLO Face Detection and Analysis"""
        try:
            # Download YOLOv8-face model if not exists
            if not os.path.exists('yolov8n-face.pt'):
                os.system('wget https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt')
            
            # Load the face detection model
            self.model = YOLO('yolov8n-face.pt')
            logging.info("YOLO face detection model loaded successfully")
            
            # Initialize face analyzer
            self.face_analyzer = FaceAnalyzer()
            
            # Set device
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logging.info(f"Using device: {self.device}")
            
        except Exception as e:
            logging.error(f"Error initializing models: {str(e)}")
            raise
    def extract_faces(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[np.ndarray]:
        """
        Extract face regions from image
        
        Args:
            image: numpy array of the image (BGR format)
            confidence_threshold: minimum confidence for face detection
            
        Returns:
            List of face region arrays
        """
        try:
            height, width = image.shape[:2]
            results = self.model(image, conf=confidence_threshold)
            face_regions = []
            
            for result in results:
                if result.boxes:
                    boxes = result.boxes
                    for box in boxes:
                        # Get coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Ensure coordinates are within image bounds
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width, x2), min(height, y2)
                        
                        # Extract face region with margin
                        margin = int((y2 - y1) * 0.2)  # 20% margin
                        face_y1 = max(0, y1 - margin)
                        face_y2 = min(height, y2 + margin)
                        face_x1 = max(0, x1 - margin)
                        face_x2 = min(width, x2 + margin)
                        
                        face_region = image[face_y1:face_y2, face_x1:face_x2].copy()
                        if face_region.size > 0:
                            face_regions.append(face_region)
            
            return face_regions
            
        except Exception as e:
            logging.error(f"Error extracting faces: {str(e)}")
            return []

    def process_image(self, image: np.ndarray, confidence_threshold: float = 0.5) -> np.ndarray:
        """
        Process image for face detection and analysis
        
        Args:
            image: numpy array of the image (BGR format)
            confidence_threshold: minimum confidence for face detection
            
        Returns:
            Processed image with detections and analysis results
        """
        try:
            # Make a copy of the image
            img = image.copy()
            height, width = img.shape[:2]
            
            # Run YOLO detection
            results = self.model(img, conf=confidence_threshold)
            
            # Process detections
            for result in results:
                if result.boxes:
                    boxes = result.boxes
                    for box in boxes:
                        # Get coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        
                        # Ensure coordinates are within image bounds
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width, x2), min(height, y2)
                        
                        # Extract face region
                        margin = int((y2 - y1) * 0.2)
                        face_y1 = max(0, y1 - margin)
                        face_y2 = min(height, y2 + margin)
                        face_x1 = max(0, x1 - margin)
                        face_x2 = min(width, x2 + margin)
                        
                        face_region = img[face_y1:face_y2, face_x1:face_x2].copy()
                        
                        if face_region.size > 0:
                            # Perform face analysis
                            name, recognition_conf = self.face_analyzer.analyze_face(face_region)
                            
                            # Draw bounding box
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            
                            # Prepare label with name and confidence
                            label = f"{name}: {recognition_conf:.2f}"
                            font = cv2.FONT_HERSHEY_DUPLEX
                            font_scale = 0.6
                            thickness = 2
                            
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
            
            return img
            
        except Exception as e:
            logging.error(f"Error processing image: {str(e)}")
            return image

    def add_face_to_database(self, name: str, face_image: np.ndarray):
        """
        Add a new face to the analysis database
        
        Args:
            name: Name of the person
            face_image: Face image array (BGR format)
        """
        try:
            self.face_analyzer.add_face(name, face_image)
            logging.info(f"Added face for {name} to database")
        except Exception as e:
            logging.error(f"Error adding face to database: {str(e)}")
            raise