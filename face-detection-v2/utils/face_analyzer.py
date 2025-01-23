from transformers import pipeline
import numpy as np
import cv2
from typing import Dict
from PIL import Image
import logging
import torch

class FaceAnalyzer:
    def __init__(self):
        try:
            # Initialize emotion classifier
            self.emotion_classifier = pipeline(
                "image-classification",
                model="dima806/facial_emotions_image_detection",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Initialize face attribute analyzer
            self.attribute_classifier = pipeline(
                "image-classification",
                model="nateraw/vit-age-classifier",
                device=0 if torch.cuda.is_available() else -1
            )
            
            logging.info("Face analyzers initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing face analyzers: {str(e)}")
            raise

    def analyze_face(self, face_image: np.ndarray) -> Dict[str, any]:
        """
        Analyze a face image for emotions and attributes.
        """
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
            
            # Get emotions
            emotion_results = self.emotion_classifier(pil_image)
            emotions = [{
                'emotion': result['label'],
                'confidence': f"{result['score']:.2f}"
            } for result in emotion_results]
            
            # Get age prediction
            age_results = self.attribute_classifier(pil_image)
            age_prediction = {
                'age_group': age_results[0]['label'],
                'confidence': f"{age_results[0]['score']:.2f}"
            }
            
            # Calculate image quality metrics
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Compile analysis results
            analysis = {
                "Emotions": emotions,
                "Age": age_prediction,
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