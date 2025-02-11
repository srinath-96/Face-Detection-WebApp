# Attempted Security System

## Project Overview
This project is a real-time face detection and emotion analysis application that uses computer vision and machine learning to detect faces and enrol it to a database for a makeshift face ID system. 

### Features
- Real-time face detection using OpenCV
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


2. Create and activate a virtual environment:
bash
For Windows:
python -m venv .venv
.venv\Scripts\activate
For macOS/Linux:
python -m venv .venv
source .venv/bin/activate


3. Install required packages:
pip install -r requirements.txt

4. Set up environment variables:
Create a .env file in the root directory and add your Hugging Face API key:
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

5. Run the application:
python app.py


## How it should Look:
<img width="775" alt="image" src="https://github.com/user-attachments/assets/490c9555-2d93-4c25-af08-5071a97f6fb7" />
<img width="775" alt="image" src="https://github.com/user-attachments/assets/4f2af2a1-59d6-4716-a73b-858970911617" />


