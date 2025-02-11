import gradio as gr
import cv2
import numpy as np
from utils.image_processor import ImageProcessor
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

class FaceRecognitionApp:
    def __init__(self):
        self.image_processor = ImageProcessor()
        logging.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    def process_webcam(self, image, confidence):
        """Process webcam feed for face recognition"""
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
            
            # Process image and detect/recognize faces
            processed_image = self.image_processor.process_image(image_bgr, confidence)
            
            # Convert back to RGB for display
            processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            
            return processed_image_rgb
            
        except Exception as e:
            logging.error(f"Error in webcam processing: {str(e)}")
            return image

    def enroll_face(self, image, name):
        """Enroll a new face in the database"""
        if image is None or not name:
            return "Please provide both image and name"
        
        try:
            # Convert RGB to BGR if necessary
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Process image to get face region
            faces = self.image_processor.extract_faces(image_bgr, confidence_threshold=0.5)
            
            if not faces:
                return "No face detected in the image"
            
            if len(faces) > 1:
                return "Multiple faces detected. Please provide an image with a single face"
            
            # Add face to database
            self.image_processor.add_face_to_database(name, faces[0])
            return f"Successfully enrolled {name}"
            
        except Exception as e:
            logging.error(f"Error enrolling face: {str(e)}")
            return f"Error enrolling face: {str(e)}"

    def create_interface(self):
        """Create the Gradio interface"""
        with gr.Blocks(title="Face Recognition System") as demo:
            gr.Markdown("# Real-time Face Recognition")
            gr.Markdown("Detect and recognize faces in real-time using deep learning")
            
            with gr.Tabs():
                # Recognition Tab
                with gr.TabItem("Recognition"):
                    with gr.Row():
                        with gr.Column():
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
                            process_button = gr.Button("Start Recognition", variant="primary")
                        
                        with gr.Column():
                            output_image = gr.Image(
                                label="Recognition Output",
                                type="numpy"
                            )
                
                # Enrollment Tab
                with gr.TabItem("Enroll New Face"):
                    with gr.Row():
                        with gr.Column():
                            enroll_camera = gr.Image(
                                label="Enrollment Camera",
                                type="numpy",
                                sources=["webcam", "upload"]
                            )
                            name_input = gr.Textbox(
                                label="Person's Name",
                                placeholder="Enter name here..."
                            )
                            enroll_button = gr.Button("Enroll Face", variant="primary")
                        
                        with gr.Column():
                            enroll_status = gr.Textbox(
                                label="Enrollment Status",
                                interactive=False
                            )
            
            # Set up event handlers
            process_button.click(
                fn=self.process_webcam,
                inputs=[camera, confidence],
                outputs=[output_image]
            )
            
            enroll_button.click(
                fn=self.enroll_face,
                inputs=[enroll_camera, name_input],
                outputs=[enroll_status]
            )
            
            # Add instructions
            gr.Markdown("""
            ### Instructions:
            
            #### Recognition:
            1. Switch to the Recognition tab
            2. Allow camera access when prompted
            3. Adjust confidence slider if needed
            4. Click 'Start Recognition' to begin
            
            #### Enrollment:
            1. Switch to the Enroll New Face tab
            2. Take a photo or upload an image
            3. Enter the person's name
            4. Click 'Enroll Face' to add to database
            
            ### Notes:
            - Higher confidence values mean stricter detection
            - Ensure good lighting for better recognition
            - For enrollment, use clear front-facing photos
            """)
        
        return demo

def main():
    try:
        # Create and launch the app
        app = FaceRecognitionApp()
        demo = app.create_interface()
        
        # Launch the app
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