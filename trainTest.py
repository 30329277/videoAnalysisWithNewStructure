import os
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pycocotools.coco import COCO
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

# 数据集路径
data_dir = "rawMaterial"  # 数据集路径
annotations_file = os.path.join(data_dir, "coco_annotations.json")

# 类别数量 (包括背景)
num_classes = 4  # 背景 + zhangsan + lisi + zhouyang (根据您的实际类别数量修改)

# 创建 COCO 对象
try:
    coco = COCO(annotations_file)
except Exception as e:
    print(f"Error loading COCO annotations: {e}")
    exit()

# 自定义变换函数
def transform(image, target):
    image = T.ToTensor()(image)  # 只对图像应用 ToTensor
    return image, target

# 创建数据集, 使用 transforms
dataset = torchvision.datasets.CocoDetection(root=data_dir, annFile=annotations_file, transforms=transform)

# 创建数据加载器
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))  # 根据GPU显存调整batch_size

# 加载预训练模型, 使用 weights 参数
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')

# 修改模型的分类器头
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# 设置设备 (GPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 设置优化器和损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# 训练循环
num_epochs = 10  # 设置训练轮数

for epoch in range(num_epochs):
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)

        # 处理 targets，提取每个目标的必要信息
        processed_targets = []
        for target in targets:
            boxes = []
            labels = []

            for t in target:  # t 是单个图像的目标列表
                box = t['bbox']  # 提取边界框
                if box[2] > box[0] and box[3] > box[1]:  # 确保边界框有效
                    boxes.append(box)
                    labels.append(t['category_id'])  # 提取类别 ID

            # 转换为张量并转换为正确的形状
            if len(boxes) > 0:  # 只有在有有效框时才继续
                boxes = torch.tensor(boxes, dtype=torch.float32).to(device)
                labels = torch.tensor(labels, dtype=torch.int64).to(device)
                processed_targets.append({
                    'boxes': boxes,
                    'labels': labels
                })

        if len(processed_targets) == 0:
            continue  # 如果没有有效框，则跳过这个批次

        # 计算损失
        loss_dict = model(images, processed_targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    # 仅在有损失的情况下打印信息
    if 'losses' in locals():  # 确保 losses 被定义
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {losses.item()}")

# 保存训练好的模型
model_save_path = "person_detection_model.pth"  # 设置模型保存路径
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")