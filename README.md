# OBJECT-TRACKING-IN-SURVEILLANCE-VIDEOS
OBJECT TRACKING IN SURVEILLANCE VIDEOS

🚀 Object Tracking in Surveillance Videos
This repository contains the Infosys Project: Object Tracking in Surveillance Videos, an implementation designed to track objects in video feeds using advanced computer vision techniques. The project utilizes OpenCV, YOLOv4, and other state-of-the-art tools to detect, classify, and track objects in real-time.

📝 Project Overview
The goal of this project is to implement object tracking for enhanced surveillance systems. By leveraging pre-trained deep learning models, the system identifies and tracks objects across video frames, providing valuable insights into motion and activity.

The key features include:

Object Detection using YOLOv3 and YOLOv4.
Real-time Tracking with custom object identifiers.
Visualization of object paths and detection boundaries.
Pixel Analysis for frame-by-frame insights.
Data Visualization using Matplotlib for histogram and RGB channel analysis.

📁 Project Structure
Installation and Setup: Instructions to set up dependencies.
Code Examples: Modular Python scripts for object detection and tracking.
Sample Videos: Real-world use case examples (user-provided video required).
Output: Processed videos with annotations, histograms, and other analysis.

🛠️ Installation
Before you begin, ensure that you have Python installed. Follow these steps to set up the project environment:

Install required Python libraries:

pip install opencv-python matplotlib pandas torch torchvision torchaudio imageai

Download YOLO models:

https://sourceforge.net/projects/imageai.mirror/files/3.0.0-pretrained/tiny-yolov3.pt/download

https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

https://github.com/pjreddie/darknet/blob/master/data/coco.names

Place the YOLO files in the models directory:


import os
os.makedirs("models", exist_ok=True)

Verify installation:


import torch
print("CUDA available:", torch.cuda.is_available())

🖥️ Usage
Clone the repository and navigate to the project directory:

git clone https://github.com/your-username/object-tracking-surveillance.git
cd object-tracking-surveillance

Run the project:


python object_tracking.py

Provide a video file in .mp4 format and ensure the path is updated in the script:


input_video_path = "path/to/your/video.mp4"

Output video with object annotations will be saved as output_video.mp4.

📊 Features and Outputs
Histogram Analysis: View pixel distribution for enhanced understanding.
RGB Channel Segmentation: Visualize separate Red, Green, and Blue channels.
Real-Time Object Tracking: Detect and track objects with unique identifiers.
Processed Video Output: Annotated video files highlighting detected objects.

🧰 Tools and Technologies
Python: The primary programming language.
OpenCV: Image and video processing.
YOLOv4: Real-time object detection.
ImageAI: Deep learning framework for computer vision.
Matplotlib & Pandas: Data visualization and analysis.
PyTorch: For deep learning-based object detection models.

📷 Sample Output
Example outputs include:

Real-time object tracking on video frames.
Annotated objects with labels and confidence scores.
Visualizations of RGB channels and pixel distributions.

FOR GETTING MORE DEFINITIVE IDEA ABOUT HOW TO EXECUTE ALL THE PROCESS GO TO MY PROJECT FILE DIRECTORY !! - https://github.com/theconsoler/OBJECT-TRACKING-IN-SURVEILLANCE-VIDEOS/blob/main/INFOSYS%20PROJECT%20-%20OBJECT%20TRACKING%20IN%20SURVEILLANCE%20VIDEOS.pdf
