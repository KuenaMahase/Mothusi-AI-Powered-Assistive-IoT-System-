import speech_recognition as sr
import time
from ultralytics import YOLO
import cv2

# Initialize speech recognizer and YOLOv8 model
recognizer = sr.Recognizer()
model = YOLO("yolov8n.pt")  # Replace with your YOLO model path
ip_webcam_url = 'http://192.168.1.14:8080/video'  # Replace with your IP Webcam URL

def detect_objects():
    """
    Function to detect objects using YOLOv8 on the IP Webcam feed.
    """
    cap = cv2.VideoCapture(ip_webcam_url)
    if not cap.isOpened():
        print(f"Error: Could not access the camera stream at {ip_webcam_url}")
        return

    print("Object detection activated.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Run YOLOv8 on the frame to detect objects
        results = model(frame)

        # Get the bounding boxes, class names, and confidence scores
        boxes = results[0].boxes  # Bounding boxes
        confidences = boxes.conf  # Confidence scores
        labels = [model.names[int(cls)] for cls in boxes.cls]  # Class names from YOLO model

        # Draw bounding boxes and labels on the frame
        for box, label, confidence in zip(boxes.xyxy, labels, confidences):
            x1, y1, x2, y2 = map(int, box)  # Coordinates of the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw the bounding box
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Label

        # Display the frame with detections
        cv2.imshow("YOLOv8 Object Detection", frame)

        # Press 'q' to quit the detection loop manually
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Object detection deactivated.")

def listen_for_commands():
    """
    Listens for specific voice commands to activate or deactivate object detection.
    """
    print("Listening for the phrase 'I need help' to activate object detection...")
    while True:
        with sr.Microphone() as source:
            try:
                recognizer.adjust_for_ambient_noise(source)  # Adjust for background noise
                print("Listening...")
                audio = recognizer.listen(source, timeout=5)
                command = recognizer.recognize_google(audio).lower()
                print(f"You said: {command}")

                if "i need help" in command:
                    print("Activating object detection...")
                    detect_objects()
                    print("Listening for 'thank you' to deactivate...")

                elif "thank you" in command:
                    print("Deactivating object detection and returning to listening mode.")
                    break

            except sr.UnknownValueError:
                print("Could not understand audio, please try again.")
            except sr.RequestError as e:
                print(f"Error with the speech recognition service: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    listen_for_commands()
