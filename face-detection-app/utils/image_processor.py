import cv2
import numpy as np
from typing import Tuple, List
import logging

def process_image(image: np.ndarray, confidence_threshold: float = 0.5) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Process an image to detect faces and draw bounding boxes.
    """
    try:
        # Make a copy of the image
        img = image.copy()
        
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=max(3, int(5 * confidence_threshold)),
            minSize=(30, 30)
        )
        
        # List to store face regions
        face_regions = []
        
        # Draw rectangles around faces and extract face regions
        for (x, y, w, h) in faces:
            # Draw rectangle
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add text label
            cv2.putText(img, f"Face", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Extract and store face region
            face_region = img[y:y+h, x:x+w].copy()
            face_regions.append(face_region)
        
        return img, face_regions
        
    except Exception as e:
        logging.error(f"Error in image processing: {str(e)}")
        return image, []