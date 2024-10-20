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
      
