from facenet_pytorch import InceptionResnetV1
import torch
import numpy as np
import cv2
from PIL import Image
import os
import pickle
from typing import Dict, List, Tuple
import logging
from sklearn.metrics.pairwise import cosine_similarity

class FaceAnalyzer:
    def __init__(self, database_path: str = 'face_database', recognition_threshold: float = 0.6):
        """
        Initialize face analysis system
        
        Args:
            database_path: Path to store face embeddings database
            recognition_threshold: Threshold for face recognition confidence
        """
        try:
            self.database_path = database_path
            self.embeddings_file = os.path.join(database_path, 'embeddings.pkl')
            self.recognition_threshold = recognition_threshold
            
            # Set device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Initialize the FaceNet model
            self.model = InceptionResnetV1(
                pretrained='vggface2',
                classify=False
            ).to(self.device).eval()
            
            # Create database directory if it doesn't exist
            os.makedirs(database_path, exist_ok=True)
            
            # Load existing embeddings if available
            self.known_embeddings = self.load_embeddings()
            
            logging.info(f"Face analysis model loaded. Known faces: {len(self.known_embeddings)}")
            
        except Exception as e:
            logging.error(f"Error initializing face analyzer: {str(e)}")
            raise

    def load_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Load stored face embeddings
        
        Returns:
            Dictionary of name to face embedding mappings
        """
        try:
            if os.path.exists(self.embeddings_file):
                with open(self.embeddings_file, 'rb') as f:
                    return pickle.load(f)
            return {}
        except Exception as e:
            logging.error(f"Error loading embeddings: {str(e)}")
            return {}

    def save_embeddings(self):
        """Save face embeddings to disk"""
        try:
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.known_embeddings, f)
            logging.info(f"Saved {len(self.known_embeddings)} embeddings to database")
        except Exception as e:
            logging.error(f"Error saving embeddings: {str(e)}")
            raise

    def preprocess_face(self, face_image: np.ndarray) -> torch.Tensor:
        """
        Preprocess face image for the model
        
        Args:
            face_image: Face image in BGR format
            
        Returns:
            Preprocessed face tensor
        """
        try:
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(face_rgb)
            
            # Resize to required size (160x160)
            pil_image = pil_image.resize((160, 160), Image.Resampling.BILINEAR)
            
            # Convert to tensor and normalize
            face_tensor = torch.FloatTensor(np.array(pil_image))
            face_tensor = face_tensor.permute(2, 0, 1)  # Convert to CxHxW
            face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension
            face_tensor = face_tensor.to(self.device)
            
            # Normalize pixel values to [-1, 1]
            face_tensor = (face_tensor - 127.5) / 128.0
            
            return face_tensor
            
        except Exception as e:
            logging.error(f"Error preprocessing face: {str(e)}")
            raise

    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Generate embedding for a face image
        
        Args:
            face_image: Face image in BGR format
            
        Returns:
            Face embedding as numpy array
        """
        try:
            # Preprocess face
            face_tensor = self.preprocess_face(face_image)
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.model(face_tensor)
                embedding = embedding.cpu().numpy()[0]
                
            return embedding
            
        except Exception as e:
            logging.error(f"Error generating embedding: {str(e)}")
            raise

    def add_face(self, name: str, face_image: np.ndarray):
        """
        Add a new face to the database
        
        Args:
            name: Name of the person
            face_image: Face image in BGR format
        """
        try:
            # Generate embedding
            embedding = self.get_embedding(face_image)
            
            # Add to database
            self.known_embeddings[name] = embedding
            
            # Save updated database
            self.save_embeddings()
            
            logging.info(f"Successfully added face for {name}")
            
        except Exception as e:
            logging.error(f"Error adding face: {str(e)}")
            raise

    def analyze_face(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        Analyze and recognize a face from the database
        
        Args:
            face_image: Face image in BGR format
            
        Returns:
            Tuple of (name, confidence)
        """
        try:
            # If no known faces, return unknown
            if not self.known_embeddings:
                return "Unknown", 0.0
            
            # Generate embedding for the input face
            embedding = self.get_embedding(face_image)
            
            # Calculate similarities with all known faces
            max_similarity = -1
            best_match = "Unknown"
            
            for name, known_embedding in self.known_embeddings.items():
                similarity = cosine_similarity(
                    embedding.reshape(1, -1), 
                    known_embedding.reshape(1, -1)
                )[0][0]
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = name
            
            # Return unknown if below threshold
            if max_similarity < self.recognition_threshold:
                return "Unknown", max_similarity
            
            return best_match, max_similarity
            
        except Exception as e:
            logging.error(f"Error in face analysis: {str(e)}")
            return "Error", 0.0

    def remove_face(self, name: str) -> bool:
        """
        Remove a face from the database
        
        Args:
            name: Name of the person to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if name in self.known_embeddings:
                del self.known_embeddings[name]
                self.save_embeddings()
                logging.info(f"Removed face for {name}")
                return True
            return False
        except Exception as e:
            logging.error(f"Error removing face: {str(e)}")
            return False

    def get_known_faces(self) -> List[str]:
        """
        Get list of all known face names
        
        Returns:
            List of names in the database
        """
        return list(self.known_embeddings.keys())