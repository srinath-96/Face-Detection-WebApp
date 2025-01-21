# utils/face_analyzer.py
import cv2
import numpy as np
from typing import Dict
from transformers import pipeline
from PIL import Image
import torch
import logging

class FaceAnalyzer:
    def __init__(self):
        try:
            # Initialize emotion classifier
            self.emotion_classifier = pipeline(
                "image-classification",  # Changed from text-classification
                model="dima806/facial_emotions_image_detection",  # Changed model
                top_k=3
            )
            
            logging.info("Face analyzer initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing face analyzer: {str(e)}")
            raise
    
    def analyze_face(self, face_image: np.ndarray) -> Dict[str, any]:
        """
        Analyze a face image and return attributes including emotion and demographics.
        """
        try:
            # Convert to grayscale for basic metrics
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Calculate basic metrics
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(face_image)
            
            # Emotion analysis
            emotion_results = self.emotion_classifier(pil_image)
            emotions = [{
                'emotion': result['label'],
                'confidence': f"{result['score']:.2f}"
            } for result in emotion_results]
            
            # Compile analysis results
            analysis = {
                "Emotions": emotions,
                "Image Quality": {
                    "Brightness": f"{brightness:.1f}",
                    "Contrast": f"{contrast:.1f}",
                    "Quality": "Good" if contrast > 30 else "Poor"
                }
            }
            
            return analysis
            
        except Exception as e:
            logging.error(f"Error in face analysis: {str(e)}")
            return {
                "Error": f"Analysis failed: {str(e)}",
                "Image Quality": {
                    "Brightness": f"{brightness:.1f}" if 'brightness' in locals() else "N/A",
                    "Contrast": f"{contrast:.1f}" if 'contrast' in locals() else "N/A",
                    "Quality": "Error"
                }
            }