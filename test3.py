import cv2
from ultralytics import YOLO
import pyttsx3  # Text-to-speech library

# Initialize the YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Set speech rate
engine.setProperty('volume', 1.0)  # Set volume (1.0 is max)

# Replace this with the IP Webcam URL (check the IP Webcam app for the URL)
ip_webcam_url = 'http://192.168.1.144:8080/video'

# Open the video stream from the IP Webcam
cap = cv2.VideoCapture(ip_webcam_url)

if not cap.isOpened():
    print(f"Error: Could not access the camera stream at {ip_webcam_url}")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Run YOLOv8 on the frame to detect objects
    results = model(frame)

    # Get the bounding boxes, class names, and confidence scores
    boxes = results[0].boxes  # Bounding boxes
    confidences = boxes.conf  # Confidence scores
    labels = [model.names[int(cls)] for cls in boxes.cls]  # Class names from the YOLO model

    # Draw bounding boxes and labels on the frame
    for box, label, confidence in zip(boxes.xyxy, labels, confidences):
        x1, y1, x2, y2 = map(int, box)  # Coordinates of the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw the bounding box
        cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Label

        # Convert to speech if confidence is 80% or higher
        if confidence >= 0.8:
            speech_text = f"I detected a {label} with {confidence:.0%} confidence."
            engine.say(speech_text)

    # Run the TTS engine (non-blocking mode to ensure smooth detection)
    engine.runAndWait()

    # Display the frame with detections
    cv2.imshow('YOLOv8 Object Detection', frame)

    # Press 'q' to quit the loop and close the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
