# Mothusi-AI-Powered-Assistive-IoT-System-
Mothusi is an AI-powered assistive assistant designed to support individuals with disabilities, especially those with visual or motor impairments. Using YOLO for object recognition, Mothusi detects and identifies objects in real-time via a camera. It processes voice commands through a microphone and provides immediate verbal feedback using a speaker, enabling users to interact with their environment more independently.
For the first Milestone,object recognition using a pre-trained AI-model, YOLOv8.

1.  Setup Instructions Prerequisites :
• Download and install  the IP webcam from Google store 
• Install Visual Studio Code  for Raspian
• Install Python(version 3)
• Install OBS Studio ( to turn laptop into Raspberry Pi screen displayer)

1. Create a virtual environment by running these commands in terminal:
    • python3 -m venv venv
    • source venv/bin/activate

2. Install the following python packages by running these commands in terminal:
    • pip install ultralytics
    • pip install opencv-python

3. Run this command to run the code:
    • python3 /file path/test2.py

NEEDED HARDWARE: Raspberry Pi 4 model B( with Raspberry OS 64-bit), Keyboard, Mouse,Monitor and mini HDMI

How Does the system Work:
    • Start the server on the IP Webcam, and then copy the http URL at the bottom of the cam into the ip_webcam_url at line 9 in the code.
    • Run the code using the command above and then move the phone’s camera in any direction. The model will start detecting the different objects around it. 

Milestone 2 - Voice Command Recognition and System Refinement
Project Name: Voice-Assisted Object Detection System
Objective

The goal of this milestone is to refine the initial design and specifications based on feedback from Milestone 1 and integrate voice command recognition capabilities into the system. The system now supports the following features:

    Voice Command Recognition:
        The system listens for specific voice commands to activate or deactivate object detection.
        Commands include:
            "I need help": Activates object detection.
            "Thank you": Deactivates object detection.

    Object Detection:
        Utilizes the YOLOv8 model to detect objects in real-time via an IP Webcam feed.
        Detects objects and overlays bounding boxes with labels and confidence scores on the video feed.

    Seamless I/O Integration:
        Incorporates voice commands through a microphone.
        Processes video from an IP Webcam.
        Provides real-time results on the screen with optional audio feedback.

Setup Instructions

    Dependencies: Ensure the following libraries are installed:

pip install ultralytics
pip install pyttsx3
pip install SpeechRecognition
pip install opencv-python
pip install opencv-python-headless

Hardware Requirements:

    A laptop or desktop with Python installed.
    A working microphone.
    An IP Webcam (use the IP Webcam app).

Code Execution:

    Replace ip_webcam_url with the actual URL of your IP Webcam feed.
    Run the script:

        python voice_object_detection.py

Code Description
Voice Command Recognition

The system listens for commands using the SpeechRecognition library.

    Command "I need help" triggers the detect_objects function.
    Command "Thank you" stops detection and returns to listening mode.

Object Detection

The YOLOv8 model processes frames from the IP Webcam feed to identify objects:

    Bounding boxes are drawn around detected objects.
    Labels and confidence scores are displayed.
    If enabled, the system provides audio feedback for high-confidence detections.

Key Features in Code
listen_for_commands

This function uses the SpeechRecognition library to:

    Adjust for ambient noise.
    Recognize voice commands from the user.
    Call the detect_objects function on the command "I need help."
    Stop detection on the command "Thank you."

detect_objects

This function performs:

    Real-time video capture from the IP Webcam feed.
    Object detection using the YOLOv8 model.
    Visual and (optional) audio feedback for detected objects.
    Allows the user to quit the loop by pressing 'q'.

Milestone Achievements

    Integrated speech recognition for command-based control of object detection.
    Ability to do text-to-speech inorder to voice out what the model has detected so that the user can hear
    Refined the input/output integration to ensure smooth communication between:
        The microphone for voice commands.
        The IP Webcam for video input.
        The YOLOv8 model for processing.
    Improved the system’s usability and user experience.

Here's an enhanced README section with the added functionality to filter objects by detecting only those with accuracy greater than 80%, along with a detailed explanation of how the code works:
How the Code Works

The Mothusi system leverages a combination of pre-trained AI models and Python libraries to perform object detection and voice command recognition seamlessly. Here's a breakdown of its functionality:

    Initialization:
        The YOLOv8 model is loaded into memory.
        The IP Webcam feed is accessed by replacing ip_webcam_url with the appropriate HTTP URL from the webcam server.
        The system initializes a listening loop for voice commands.

    Voice Command Recognition:
        The system uses the SpeechRecognition library to listen for specific commands:
            "I need help": Activates the object detection functionality.
            "Thank you": Deactivates object detection and returns to idle mode.
        Commands are processed through a microphone, with the system adjusting for ambient noise to enhance recognition accuracy.

    Object Detection:
        The YOLOv8 model processes video frames from the IP Webcam feed.
        Objects are detected in real-time, with bounding boxes, labels, and confidence scores displayed on the video feed.

    Object Filtering:
        The system filters detections to display only objects with a confidence score of 80% or higher, ensuring high-quality and reliable results.
        Objects below the confidence threshold are ignored, reducing visual and auditory clutter.

    Audio Feedback:
        For high-confidence detections (greater than 80%), the system optionally announces detected objects using the pyttsx3 text-to-speech engine.
        This feature provides real-time verbal feedback, enhancing accessibility for users with visual impairments.

    User Interaction:
        The system runs continuously until the user presses the 'q' key to quit.

Example Code Snippet: Object Filtering

Below is a key snippet showing how the system filters objects by confidence score:

from ultralytics import YOLO
import cv2
import pyttsx3

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# Set the confidence threshold
CONFIDENCE_THRESHOLD = 0.8

# Initialize text-to-speech engine
engine = pyttsx3.init()

def detect_objects(video_feed_url):
    cap = cv2.VideoCapture(video_feed_url)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run object detection
        results = model(frame)

        for result in results:
            for obj in result.boxes.data:
                confidence = float(obj[-1])  # Extract confidence score
                label = result.names[int(obj[-2])]  # Extract object label

                # Filter objects by confidence
                if confidence >= CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, obj[:4])  # Extract bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Provide audio feedback
                    engine.say(f"Detected {label} with confidence {confidence:.2f}")
                    engine.runAndWait()

        # Display the video feed
        cv2.imshow("Object Detection", frame)

        # Quit the loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

Enhanced System Features

    Accuracy Filtering:
        The system detects only objects with confidence scores above 80%, ensuring reliable feedback.
    Real-Time Feedback:
        Displays bounding boxes and labels on the screen.
        Announces high-confidence detections verbally.


      
