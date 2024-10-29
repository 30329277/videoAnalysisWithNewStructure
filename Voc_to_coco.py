import os
import json
from xml.etree import ElementTree
import cv2
from pycocotools import mask

# 定义路径
raw_material_dir = "rawMaterial"
output_json_file = "rawMaterial/coco_annotations.json"

# COCO 数据结构
coco = {
    "info": {"description": "Person Detection Dataset", "version": "1.0"},
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": [
        {"supercategory": "person", "id": 1, "name": "zhouyang"},
        {"supercategory": "person", "id": 2, "name": "wangfuliang"},
        # ... 添加其他类别
    ],
}

image_id = 1
annotation_id = 1

# 遍历 XML 文件
for filename in os.listdir(raw_material_dir):
    if filename.endswith(".xml"):
        xml_path = os.path.join(raw_material_dir, filename)
        image_path = os.path.join(raw_material_dir, filename[:-4] + ".jpg")  # 假设图片是 jpg 格式

        # 解析 XML 文件
        tree = ElementTree.parse(xml_path)
        root = tree.getroot()

        # 获取图像信息
        img = cv2.imread(image_path)
        height, width, _ = img.shape

        image_info = {
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": filename[:-4] + ".jpg",  # 图片文件名
            "license": None,  # 可选
            "flickr_url": None,  # 可选
            "coco_url": None,  # 可选
            "date_captured": None,  # 可选
        }
        coco["images"].append(image_info)

        # 获取标注信息
        for obj in root.findall("object"):
            name = obj.find("name").text
            category_id = next((c["id"] for c in coco["categories"] if c["name"] == name), None)
            if category_id is None:
                print(f"Warning: Category '{name}' not found in categories.")
                continue

            bbox = [int(float(obj.find("bndbox").find(coord).text)) for coord in ["xmin", "ymin", "xmax", "ymax"]]
            xmin, ymin, xmax, ymax = bbox
            width = xmax - xmin
            height = ymax - ymin

            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [], #  对于目标检测，segmentation 可以为空列表
                "area": width * height,
                "bbox": [xmin, ymin, width, height],
                "iscrowd": 0,  # 0 表示单个对象，1 表示人群
            }
            coco["annotations"].append(annotation)
            annotation_id += 1

        image_id += 1

# 保存 JSON 文件
with open(output_json_file, "w") as f:
    json.dump(coco, f, indent=4)

print(f"COCO annotations saved to {output_json_file}")