from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
import torch
import torchvision
import cv2
import pandas
from PIL import Image
import numpy as np
import math
from datetime import datetime

app = Flask(__name__)

# Check if CUDA is available and use it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load the trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
model.to(device)
model.eval()
print("Model loaded from best.pt")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        processed_video_path = process_video(file_path)
        return redirect(url_for('show_video', filename=os.path.basename(processed_video_path)))
    return redirect(request.url)

@app.route('/video/<filename>')
def show_video(filename):
    return render_template('video.html', filename=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('processed_videos', filename)


# Function to scale the frame while maintaining aspect ratio
def scale_frame(frame, max_width, max_height):
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h))

def euc_distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def detect_movement(center_points, start_point, velocity_threshold=5, dist_threshold=20):
    if len(center_points) <= velocity_threshold:
        return "Start"
    
    velocity = center_points[-1][1] - center_points[-2][1]
    if abs(velocity) < velocity_threshold:
        return "Static"
    elif is_racking(center_points, velocity_threshold):
        return "Racking"
    elif euc_distance(start_point, center_points[-velocity_threshold]) < euc_distance(start_point, center_points[-1]):
        return "Away"
    else:
        return "Towards"

def is_racking(center_points, time_threshold=8):
    x_diff = center_points[-1][0] - center_points[-time_threshold][0]
    y_diff = center_points[-1][1] - center_points[-time_threshold][1]
    return True if abs(x_diff) > abs(y_diff) else False


def process_video(file_path):
    cap = cv2.VideoCapture(file_path)
    vid_fps = int(cap.get(cv2.CAP_PROP_FPS))

    tracker = cv2.legacy.TrackerCSRT_create()
    ret, frame = cap.read()
    if not ret:
        return "Failed to read video"

    frame = scale_frame(frame, 1500, 1000)
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = model(frame_pil)
    if len(results.pandas().xyxy[0]) > 0:
        bbox = results.pandas().xyxy[0]
        x_min = bbox['xmin'].iloc[0]
        y_min = bbox['ymin'].iloc[0]
        x_max = bbox['xmax'].iloc[0]
        y_max = bbox['ymax'].iloc[0]
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        tracker.init(frame, bbox)
        center_x = int(bbox[0] + bbox[2] / 2)
        center_y = int(bbox[1] + bbox[3] / 2)
        start_point = (center_x, center_y)
    else:
        return "No barbell detected in the first frame"

    center_points = []
    rep_threshold = 2
    min_distance_threshold = 50
    rep_count = 0
    frame_count = 0
    fastest_velocity = 0
    velocity_loss = 0
    is_significant_move = False
    rep_done = True

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    processed_video_path = os.path.join('processed_videos', os.path.basename(file_path))
    out = cv2.VideoWriter(processed_video_path, fourcc, vid_fps, (frame.shape[1], frame.shape[0]))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame = scale_frame(frame, 1500, 1000)
        ret, bbox = tracker.update(frame)
        if ret:
            center_x = int(bbox[0] + bbox[2] / 2)
            center_y = int(bbox[1] + bbox[3] / 2)
            center_points.append((center_x, center_y))
            curr_phase = detect_movement(center_points, start_point)
            if curr_phase == "Away" and rep_done == True:
                peak_point = center_points[-1]
                rep_start_frame = frame_count
                rep_done = False
            if not rep_done and not is_significant_move:
                vertical_distance = abs(center_points[-1][1] - peak_point[1])
                if vertical_distance > min_distance_threshold:
                    is_significant_move = True
            if is_significant_move and not rep_done and peak_point is not None:
                vertical_distance = abs(center_points[-1][1] - peak_point[1])
                if vertical_distance < min_distance_threshold:
                    rep_count += 1
                    rep_end_frame = frame_count
                    rep_time = (rep_end_frame - rep_start_frame) / vid_fps
                    velocity = vertical_distance / rep_time
                    if velocity > fastest_velocity:
                        fastest_velocity = velocity
                    velocity_loss = (fastest_velocity - velocity) / fastest_velocity * 100
                    rep_done = True
                    is_significant_move = False
        else:
            break

        # Draw lines connecting the center points
        for i in range(1, len(center_points)):
            cv2.line(frame, center_points[i - 1], center_points[i], (0, 255, 0), 2)
        
        cv2.putText(frame, f"Rep Count: {rep_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Velocity Loss: {velocity_loss:.2f}%", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    return processed_video_path