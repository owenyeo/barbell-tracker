import torch
import torchvision
import cv2
import pandas
from PIL import Image
import numpy as np

# Check if CUDA is available and use it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load the trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
model.to(device)
model.eval()
print("Model loaded from best.pt")

# Function to scale the frame while maintaining aspect ratio
def scale_frame(frame, max_width, max_height):
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h))

# Initialize video capture for the webcam or a video file
cap = cv2.VideoCapture("test.mp4")  

# Initialize tracker
tracker = cv2.legacy.TrackerCSRT_create()

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Failed to read from webcam")
    cap.release()
    exit()

# Scale the frame for display
frame = scale_frame(frame, 1500, 1000)

frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
results = model(frame_pil)
print(results.pandas().xyxy[0])

# Initialize tracker with the bounding box from the detection
if len(results.pandas().xyxy[0]) > 0:
    bbox = results.pandas().xyxy[0]
    x_min = bbox['xmin'].iloc[0]
    y_min = bbox['ymin'].iloc[0]
    x_max = bbox['xmax'].iloc[0]
    y_max = bbox['ymax'].iloc[0]
    bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
    tracker.init(frame, bbox)
else:
    print("No barbell detected in the first frame")
    cap.release()
    exit()

centre_points = []

# Track the barbell in the live feed
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Scale the frame for display
    frame = scale_frame(frame, 1500, 1000)
    
    # Update the tracker
    ret, bbox = tracker.update(frame)
    print(bbox)
    
    if ret:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)

        # Calculate the center point of the bounding box
        center_x = int(bbox[0] + bbox[2] / 2)
        center_y = int(bbox[1] + bbox[3] / 2)
        centre_points.append((center_x, center_y))

        # Draw lines connecting the center points
        for i in range(1, len(centre_points)):
            cv2.line(frame, centre_points[i - 1], centre_points[i], (0, 255, 0), 2)

    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Tracking", frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
