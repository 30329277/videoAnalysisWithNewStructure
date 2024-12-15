import os
import torch
import torchvision
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import xml.etree.ElementTree as ET
import numpy as np
import torch.multiprocessing as mp
import sys
import tqdm

def collate_fn(batch):
    return tuple(zip(*batch))

class ObjectDataset(Dataset):
    def __init__(self, data_dir, object_name, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.object_name = object_name

        self.image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.annotation_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.xml')]

        self.data = []
        for img_file in self.image_files:
            annotation_file = img_file[:-4] + '.xml'
            if annotation_file in self.annotation_files:
                if self.check_object_in_xml(os.path.join(data_dir, annotation_file)):
                    self.data.append((img_file, annotation_file))

    def check_object_in_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            label = obj.find('name').text
            if label == self.object_name:
                return True
        return False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, annotation_name = self.data[idx]
        img_path = os.path.join(self.data_dir, img_name)
        annotation_path = os.path.join(self.data_dir, annotation_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            label = obj.find('name').text
            if label == self.object_name:
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(1)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)

        if self.transform:
            img = self.transform(img)

        return img, target

def get_object_names(data_dir):
    object_names = set()
    for filename in os.listdir(data_dir):
        if filename.lower().endswith('.xml'):
            try:
                tree = ET.parse(os.path.join(data_dir, filename))
                root = tree.getroot()
                for obj in root.findall('object'):
                    object_names.add(obj.find('name').text)
            except ET.ParseError as e:
                print(f"Error parsing XML file {filename}: {e}")
                continue
    return list(object_names)

def train_model(data_dir, object_name, num_epochs=10, batch_size=2, learning_rate=0.005, momentum=0.9, num_workers=0):
    model_path = f"{object_name}_detector_model.pth"

    dataset = ObjectDataset(data_dir, object_name, transform=torchvision.transforms.ToTensor())
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)

    if os.path.exists(model_path):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2) # num_classes is always 2 (background + object)
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        print(f"Loaded pretrained model from {model_path}")
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        print("Initialized model with pretrained weights from torchvision")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        with tqdm.tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as tepoch:
            for images, targets in tepoch:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                epoch_loss += losses.item()

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                tepoch.set_postfix(loss=losses.item())

        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {epoch_loss / len(data_loader)}")

    torch.save(model.state_dict(), model_path)
    print(f"模型训练完成并保存为 {model_path}")
    return model

if __name__ == '__main__':
    mp.freeze_support()

    data_dir = r"D:\PythonProject\objectA model03" #  Replace with your data directory
    num_workers = 0

    object_names = get_object_names(data_dir)
    if not object_names:
        print("No object names found in XML files. Exiting.")
        sys.exit(1)

    for object_name in object_names:
        print(f"Training model for {object_name}")
        trained_model = train_model(data_dir, object_name, num_workers=num_workers)