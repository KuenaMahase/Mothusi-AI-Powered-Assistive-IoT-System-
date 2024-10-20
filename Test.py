import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Load the image
image_path = '/home/kuenamahase/Downloads/dog.jpg'  # Update this path to your image
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image at {image_path}. Please check the file path.")
else:
    # Run object detection
    results = model(image)

    # Get the bounding boxes, class names, and confidence scores
    boxes = results[0].boxes  # Bounding boxes
    confidences = boxes.conf  # Confidence scores
    labels = [model.names[int(cls)] for cls in boxes.cls]  # Class names from model

    # Draw bounding boxes and labels on the image
    for box, label, confidence in zip(boxes.xyxy, labels, confidences):
        x1, y1, x2, y2 = map(int, box)  # Coordinates of the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the image with detections
    cv2.imshow('YOLOv8 Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the output image
    output_image_path = 'output_image.jpg'
    cv2.imwrite(output_image_path, image)
    print(f"Output saved to {output_image_path}")
