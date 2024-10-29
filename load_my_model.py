import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import ToTensor
from PIL import Image
import matplotlib.pyplot as plt

# 设备设置
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Device:", device)

# 加载预训练模型并修改分类器头
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
in_features = model.roi_heads.box_predictor.cls_score.in_features
num_classes = 4  # 根据您的类别数量
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# 加载模型权重，设置 weights_only=True
try:
    model.load_state_dict(torch.load("person_detection_model.pth", weights_only=True))
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

model.to(device)
model.eval()  # 设置为评估模式

# 进行推理的函数
def predict(image_path):
    try:
        # 加载并处理图像
        image = Image.open(image_path)
        print(f"Image {image_path} loaded successfully.")
        image_tensor = ToTensor()(image).unsqueeze(0).to(device)  # 增加批次维度

        with torch.no_grad():  # 关闭梯度计算
            predictions = model(image_tensor)

        return predictions
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

# 使用示例
image_path = r"rawMaterial\03331_017.jpg"  # 确保路径正确
print(f"Using image path: {image_path}")
predictions = predict(image_path)

if predictions is not None:
    print("Predictions made successfully.")
else:
    print("No predictions were made.")

    # 显示预测结果的函数
def display_predictions(image_path, predictions):
    image = Image.open(image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # 处理预测结果
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):
        if score > 0.5:  # 只显示高于阈值的预测
            x_min, y_min, x_max, y_max = box
            plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                                fill=False, color='red', linewidth=2))
            plt.text(x_min, y_min, f'Label: {label}, Score: {score:.2f}', 
                     color='red', fontsize=12)

    plt.axis('off')
    plt.show()

# 调用显示函数
if predictions is not None:
    display_predictions(image_path, predictions)