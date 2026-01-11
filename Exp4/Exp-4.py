# detect_bike.py
import cv2
import os
from ultralytics import YOLO

# 加载预训练模型（COCO）
model = YOLO('yolov5s.pt')  # 使用 yolov5s.pt（小模型）

# COCO 类别中 "bicycle" 的索引
bike_class_idx = 1  # 在COCO数据集中，bicycle的类别ID是1

# 图像路径
image_path = 'test.jpg'  # 当前目录下图片
output_path = 'result_with_bikes.jpg'

# 读取图像
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

# 执行检测
results = model(img, conf=0.7)  # conf参数设置置信度阈值

# 获取检测结果
result = results[0]  # 获取第一个(也是唯一一个)结果
original_img = result.orig_img.copy()  # 保存原始图像的副本用于绘制

# 处理检测结果
if result.boxes is not None and len(result.boxes):
    # 遍历所有检测到的对象
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy()  # 获取边界框坐标

        # 只处理自行车
        if class_id == bike_class_idx:
            x1, y1, x2, y2 = map(int, xyxy)
            label = f'bicycle {confidence:.2f}'

            # 绘制边界框
            cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 绘制标签
            cv2.putText(original_img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# 保存结果
cv2.imwrite(output_path, original_img)
print(f"Detection result saved to {output_path}")

# 可选：显示结果
cv2.imshow('Bicycle Detection', original_img)
cv2.waitKey(0)
cv2.destroyAllWindows()