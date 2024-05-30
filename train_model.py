import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import torch.optim as optim
from torchvision.transforms import functional as F

def collate_fn(batch):
    images, targets = zip(*batch)
    
    max_width = max(image.shape[2] for image in images)
    max_height = max(image.shape[1] for image in images)
    
    padded_images = []
    padded_targets = []
    
    for image, target in zip(images, targets):
        pad_right = max_width - image.shape[2]
        pad_bottom = max_height - image.shape[1]
        
        padded_image = F.pad(image, (0, 0, pad_right, pad_bottom))
        padded_images.append(padded_image)
        
        boxes = target["boxes"].clone()
        boxes[:, [0, 2]] += 0
        boxes[:, [1, 3]] += 0
        
        padded_targets.append({
            "boxes": boxes,
            "labels": target["labels"]
        })
    
    padded_images = torch.stack(padded_images)
    
    return padded_images, padded_targets

class BarbellDataset(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None):
        self.img_dir = img_dir
        self.annotations = pd.read_csv(annotations_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_info = self.annotations.iloc[idx]
        img_path = os.path.join(self.img_dir, img_info['filename'])
        img = Image.open(img_path).convert("RGB")

        boxes = [[img_info['xmin'], img_info['ymin'], img_info['xmax'], img_info['ymax']]]
        boxes = torch.tensor(boxes, dtype=torch.float32)

        labels = torch.ones((len(boxes),), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transform:
            img = self.transform(img)

        return img, target

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='COCO_V1')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

num_classes = 2
model = get_model(num_classes)

# Create the dataset and dataloader
dataset = BarbellDataset(img_dir='raw', annotations_file='annotations.csv', transform=ToTensor())
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn)

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, targets in dataloader:
        images = list(image for image in images)
        targets = [{'boxes': target['boxes'], 'labels': target['labels']} for target in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {losses.item()}')

torch.save(model.state_dict(), 'barbell_detector.pth')
print("Model saved to barbell_detector.pth")
