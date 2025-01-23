import gradio as gr
import cv2
import numpy as np
from utils.image_processor import ImageProcessor
from utils.face_analyzer import FaceAnalyzer
import os
from dotenv import load_dotenv
import logging
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

def format_analysis(analysis_dict):
    """Format analysis results for display"""
    text = ""
    for key, value in analysis_dict.items():
        if key == "Emotions":
            text += f"\n{key}:\n"
            for emotion in value:
                text += f"  - {emotion['emotion']}: {emotion['confidence']}\n"
        elif key == "Age":
            text += f"\n{key}:\n"
            text += f"  - Group: {value['age_group']}\n"
            text += f"  - Confidence: {value['confidence']}\n"
        elif isinstance(value, dict):
            text += f"\n{key}:\n"
            for sub_key, sub_value in value.items():
                text += f"  - {sub_key}: {sub_value}\n"
        else:
            text += f"{key}: {value}\n"
    return text

class FaceDetectionApp:
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.face_analyzer = FaceAnalyzer()
        logging.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    def process_webcam(self, image, confidence):
        """Process webcam feed and detect faces"""
        try:
            if image is None:
                return None, "No image received"
            
            # Ensure image is in the correct format (BGR)
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert RGB to BGR if necessary
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                logging.error(f"Unexpected image format: {image.shape}")
                return image, "Error: Invalid image format"
            
            # Process image and detect faces
            processed_image, faces = self.image_processor.process_image(image_bgr, confidence)
            
            # Convert back to RGB for display
            processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            
            # Analyze faces if any were detected
            analysis_text = ""
            if len(faces) > 0:
                analysis_text += f"Found {len(faces)} face(s)\n"
                for i, face in enumerate(faces, 1):
                    analysis_text += f"\n=== Face {i} ===\n"
                    # Convert face to RGB for analysis
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    analysis = self.face_analyzer.analyze_face(face_rgb)
                    analysis_text += format_analysis(analysis)
            else:
                analysis_text = "No faces detected"
            
            return processed_image_rgb, analysis_text
            
        except Exception as e:
            logging.error(f"Error in webcam processing: {str(e)}")
            return image, f"Error processing image: {str(e)}"

    def create_interface(self):
        """Create the Gradio interface"""
        with gr.Blocks(title="Face Detection & Analysis") as demo:
            gr.Markdown("# Face Detection & Analysis with YOLOv8")
            gr.Markdown("Real-time face detection and analysis using state-of-the-art models")
            
            with gr.Row():
                with gr.Column():
                    # Input components
                    camera = gr.Image(
                        label="Camera Input",
                        type="numpy",
                        sources=["webcam"],
                        streaming=True
                    )
                    confidence = gr.Slider(
    minimum=0.2,
    maximum=0.9,
    value=0.5,
    step=0.05,
    label="Detection Confidence"
)
                    process_button = gr.Button("Process Image", variant="primary")
                
                with gr.Column():
                    # Output components
                    output_image = gr.Image(
                        label="Processed Image",
                        type="numpy"
                    )
                    analysis_text = gr.Textbox(
                        label="Analysis Results",
                        lines=10,
                        max_lines=20
                    )
            
            # Process button click event
            process_button.click(
                fn=self.process_webcam,
                inputs=[camera, confidence],
                outputs=[output_image, analysis_text]
            )
            
            gr.Markdown("""
            ### Features:
            - Advanced Face Detection using YOLOv8
            - Emotion Analysis
            - Age Group Estimation
            - Image Quality Assessment
            
            ### Instructions:
            1. Allow camera access when prompted
            2. Adjust confidence slider if needed (higher = stricter detection)
            3. Click 'Process Image' to analyze the current frame
            4. View results in real-time
            """)
        
        return demo

def main():
    try:
        # Create and launch the app
        app = FaceDetectionApp()
        demo = app.create_interface()
        
        # Launch the app with updated parameters
        demo.launch(
            share=False,          # Set to True to create a public link
            server_name="0.0.0.0",  # Allows external access
            server_port=7860,     # Default Gradio port
            quiet=True            # Reduces console output
        )
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main()