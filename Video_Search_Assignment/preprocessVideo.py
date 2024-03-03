import os
import cv2
import numpy as np
import torch
import csv
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image

# Make sure PyTorch is in evaluation mode
torch.set_grad_enabled(False)

# Load the pre-trained model
model = fasterrcnn_resnet50_fpn(weight=True)
model.eval()  # Set the model to evaluation mode

# Define the COCO classes
COCO_CLASSES = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# List of videos to process
video_paths = [
    '/Users/sandeebadhikari/Documents/cs370-assignments/Video_Search_Assignment/Downloads/YouTube-Videos/How Green Roofs Can Help Cities  NPR.mp4',
    '/Users/sandeebadhikari/Documents/cs370-assignments/Video_Search_Assignment/Downloads/YouTube-Videos/What Does High-Quality Preschool Look Like  NPR Ed.mp4',
    '/Users/sandeebadhikari/Documents/cs370-assignments/Video_Search_Assignment/Downloads/YouTube-Videos/Why It’s Usually Hotter In A City  Lets Talk  NPR.mp4'
]

# Parameters for frame sampling and resizing
frame_sample_rate = 30  # How often to sample a frame
resize_width = 224
resize_height = 224

# Directory where you want to save the detection results
save_directory = '/Users/sandeebadhikari/Documents/cs370-assignments/Video_Search_Assignment/Detection_Results/'

# Ensure the save directory exists, create it if it does not
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Function to process a single video and perform object detection
def process_and_detect(video_path, frame_sample_rate, resize_dims, model):
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return None

    frame_count = 0
    detections = []

    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame at specified rate
        if frame_count % frame_sample_rate == 0:
            # Convert frame to PIL Image and preprocess
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            transform = T.Compose([T.Resize(resize_dims), T.ToTensor()])
            transformed_image = transform(pil_image).unsqueeze(0)

            # Perform detection
            prediction = model(transformed_image)

            # Extract detection data
            boxes = prediction[0]['boxes']
            labels = prediction[0]['labels']
            scores = prediction[0]['scores']

            for box, label, score in zip(boxes, labels, scores):
                if score >= 0.5:
                    label_id = int(label.item())
                    # Check if the label ID is within the bounds of COCO_CLASSES
                    if label_id < len(COCO_CLASSES):
                        class_name = COCO_CLASSES[label_id]
                    else:
                        class_name = 'Unknown'  # Handle out-of-bound label IDs
                        
                    detections.append([
                        os.path.basename(video_path),  # vidId
                        frame_count,  # frameNum
                        frame_count / cap.get(cv2.CAP_PROP_FPS),  # timestamp (sec)
                        label_id,  # detectedObjId
                        class_name,  # detectedObjClass
                        float(score.item()),  # confidence
                        box.tolist()  # bbox info
                    ]) 
        frame_count += 1

    cap.release()
    return detections

# Function to save detections to a CSV file
def save_detections(detections, save_path):
    with open(save_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['vidId', 'frameNum', 'timestamp', 'detectedObjId', 'detectedObjClass', 'confidence', 'bbox info'])
        writer.writerows(detections)
        
# Process each video and save detection results
for video_path in video_paths:
    print(f"Processing video: {video_path}")
    detections = process_and_detect(video_path, frame_sample_rate, (resize_width, resize_height), model)
    if detections:
        save_path = os.path.join(save_directory, os.path.basename(video_path).replace('.mp4', '_detections.csv'))
        save_detections(detections, save_path)
        print(f"Detections saved to {save_path}")
    else:
        print("No detections were found.")
