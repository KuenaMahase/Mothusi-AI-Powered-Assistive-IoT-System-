import cv2
from ultralytics import YOLO

# Initialize the YOLOv8 model (make sure 'yolov8n.pt' is in the same directory or provide the correct path)
model = YOLO('yolov8n.pt')

# Replace this with the IP Webcam URL (check IP Webcam app for the URL)
# Example: 'http://192.168.x.x:8080/video'
ip_webcam_url = 'http://192.168.1.148:8080/video'

# Open the video stream from IP Webcam
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
    labels = [model.names[int(cls)] for cls in boxes.cls]  # Class names from YOLO model

    # Draw bounding boxes and labels on the frame
    for box, label, confidence in zip(boxes.xyxy, labels, confidences):
        x1, y1, x2, y2 = map(int, box)  # Coordinates of the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw the bounding box
        cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Label

    # Display the frame with detections
    cv2.imshow('YOLOv8 Object Detection', frame)

    # Press 'q' to quit the loop and close the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
