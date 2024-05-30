import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import ToTensor
import cv2
from PIL import Image
import numpy as np

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='COCO_V1')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Load the trained model
num_classes = 2  # Background and barbell
model = get_model(num_classes)
model.load_state_dict(torch.load('barbell_detector.pth'))
model.eval()
print("Model loaded from barbell_detector.pth")

def detect_barbell(frame, model):
    transform = ToTensor()
    img = transform(frame).unsqueeze(0)
    with torch.no_grad():
        prediction = model(img)
    return prediction

# Video path
video_path = 'test.mp4'

# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Initialize tracker
tracker = cv2.legacy.TrackerCSRT_create()

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Failed to read video")
    cap.release()
    exit()

# Detect the barbell in the first frame
frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
predictions = detect_barbell(frame_pil, model)
print(predictions)

# Initialize tracker with the bounding box from the detection
if len(predictions[0]['boxes']) > 0:
    bbox = predictions[0]['boxes'][0].int().tolist()
    bbox = tuple(bbox)
    tracker.init(frame, bbox)
else:
    print("No barbell detected in the first frame")
    cap.release()
    exit()

# Track the barbell
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Update the tracker
    ret, bbox = tracker.update(frame)
    
    if ret:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
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