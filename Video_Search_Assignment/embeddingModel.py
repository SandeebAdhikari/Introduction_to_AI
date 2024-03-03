import cv2
import torch
import psycopg2
import io
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize  # Corrected import here
from torch.utils.data import DataLoader
from PIL import Image
from autoencoder import ConvAutoencoder
from autoencoder import InMemoryCroppedObjectDataset
from autoencoder import train_autoencoder

# Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detection_model = fasterrcnn_resnet50_fpn(weight=True).to(device).eval()
autoencoder = ConvAutoencoder().to(device).eval()  # Assuming training is done, and we're in inference mode


# COCO classes
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

# Transformation for detected objects before passing to the autoencoder
transform = T.Compose([
    T.Resize((224, 224)),  # Match autoencoder input size
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ImageNet norms
])


frame_sample_rate = 30  # How often to sample a frame
def process_video_and_extract_embeddings(video_path, detection_model, autoencoder, transform, device):
    cap = cv2.VideoCapture(video_path)
    cropped_images =[]
    embeddings = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_sample_rate == 0:
            # Convert frame to PIL for easier processing and detection
            pil_image = Image.fromarray(frame)
            img_tensor = T.ToTensor()(pil_image).unsqueeze(0).to(device)

            # Object detection
            with torch.no_grad():
                predictions = detection_model(img_tensor)
                
            
            # Loop through detections
            for i, (box, score, label) in enumerate(zip(predictions[0]['boxes'], predictions[0]['scores'], predictions[0]['labels'])):
                if score >= 0.5:  # Confidence threshold
                    class_name = COCO_CLASSES[label]
                    if class_name in COCO_CLASSES:
                        # Crop detected object
                        box = [round(b.item()) for b in box]
                        cropped_obj = pil_image.crop((box[0], box[1], box[2], box[3]))
                            
                        cropped_images.append(cropped_obj)
                        cropped_obj_tensor = transform(cropped_obj).unsqueeze(0).to(device)

                        # Pass cropped object through autoencoder to get embedding
                        with torch.no_grad():
                            _, embedding = autoencoder(cropped_obj_tensor)
                            embeddings.append(embedding.squeeze().cpu().numpy())
    cap.release()
    return embeddings, cropped_images

video_path = [
    '/Users/sandeebadhikari/Documents/cs370-assignments/Video_Search_Assignment/Downloads/YouTube-Videos/How Green Roofs Can Help Cities  NPR.mp4',
    '/Users/sandeebadhikari/Documents/cs370-assignments/Video_Search_Assignment/Downloads/YouTube-Videos/What Does High-Quality Preschool Look Like  NPR Ed.mp4',
    '/Users/sandeebadhikari/Documents/cs370-assignments/Video_Search_Assignment/Downloads/YouTube-Videos/Why It’s Usually Hotter In A City  Lets Talk  NPR.mp4'
]

for path in video_path:
    embeddings, cropped_images = process_video_and_extract_embeddings(path, detection_model, autoencoder, transform, device)
 

print(len(cropped_images), len(embeddings))


# Now, use all_cropped_images for dataset creation
dataset = InMemoryCroppedObjectDataset(cropped_images, transform=transform)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize your autoencoder and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = ConvAutoencoder().to(device)

# Optimizer and loss criterion
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001) 
criterion = torch.nn.MSELoss()

# Start training
train_autoencoder(autoencoder, data_loader, optimizer, criterion, epochs=5, device=device)

def image_to_byte_array(image:Image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=image.format)
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr

conn = psycopg2.connect("dbname=citus user=citus password='Starter$05")
cur = conn.cursor()

for cropped_image, embedding in zip(cropped_images, embeddings):
    # Convert the PIL Image to bytes
    image_data = image_to_byte_array(cropped_image)
    # Insert into the database
    cur.execute("INSERT INTO image_embeddings (image_data, embedding) VALUES (%s, %s)", (image_data, embedding))

conn.commit()
cur.close()
conn.close()