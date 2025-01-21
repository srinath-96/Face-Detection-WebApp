# Real-time Face Detection and Emotion Analysis

## Project Overview
This project is a real-time face detection and emotion analysis application that uses computer vision and machine learning to detect faces and analyze emotions in real-time through your webcam. The application combines OpenCV for face detection and Hugging Face's transformers for emotion analysis.

### Features
- Real-time face detection using OpenCV
- Emotion analysis using pre-trained deep learning models
- Image quality assessment (brightness, contrast)
- User-friendly interface built with Gradio
- Adjustable detection confidence settings
- Real-time processing and analysis

## Technologies Used
- Python 3.10+
- OpenCV for face detection
- Hugging Face Transformers for emotion analysis
- Gradio for the web interface
- PyTorch for deep learning
- NumPy for numerical operations

## Prerequisites
Before running the application, ensure you have:
1. Python 3.10 or higher installed
2. Webcam access
3. Hugging Face API token (get it from https://huggingface.co/settings/tokens)
4. Git (for cloning the repository)

## Installation Steps

1. Clone the repository:
bash
git clone https://github.com/srinath-96/Face-Detection-WebApp.git

cd face-detection-app


3. Create and activate a virtual environment:
bash
For Windows:
python -m venv .venv

.venv\Scripts\activate

For macOS/Linux:

python -m venv .venv

source .venv/bin/activate


5. Install required packages:
6. 
pip install -r requirements.txt

7. Set up environment variables:
Create a .env file in the root directory and add your Hugging Face API key:

HUGGINGFACE_API_KEY=your_huggingface_api_key_here

9. Run the application:
python app.py





## Usage Instructions
1. After running `python app.py`, the application will open in your default web browser
2. Allow camera access when prompted
3. Adjust the confidence slider if needed (higher values for more strict face detection)
4. Click the "Process Image" button to analyze the current frame
5. View results in real-time:
   - Processed image with face detection boxes
     
   - Emotion analysis results
     
   - Image quality metrics

## Troubleshooting

### Common Issues and Solutions:

1. Camera not working:
   - Check browser permissions
   - Ensure no other application is using the camera
   - Try refreshing the page

2. Installation errors:
   - Ensure you're using Python 3.10+
   - Try creating a new virtual environment
   - Update pip before installing requirements

3. Model download issues:
   - Check your internet connection
   - Verify your Hugging Face API token
   - Ensure enough disk space for models

4. Performance issues:
   - Close other resource-intensive applications
   - Reduce the camera resolution if needed
   - Consider using a GPU for better performance

## Development and Contribution

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request


## Acknowledgments
- OpenCV for computer vision capabilities
- Hugging Face for transformer models
- Gradio team for the web interface framework


## Version History(will be working on adding more features and using more models)
- v1.0.0 (2024-01-20)
  - Initial release
  - Basic face detection and emotion analysis
  - Real-time processing capabilities
