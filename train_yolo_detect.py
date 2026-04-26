from ultralytics import YOLO

# 加载预训练的YOLO11目标检测模型
model = YOLO("yolo11s.pt")

# 训练模型（调整batch）
train_results = model.train(
    data="data_detect.yaml",  # 数据集配置文件
    epochs =200,            # 训练轮次
    imgsz=640,              # 图像尺寸
    device=0,               # 核心：启用GPU训练
    batch=4,                # 减小Batch Size（适配GPU显存，可按需调整）
    workers=0               # 减少数据加载线程（避免CPU占用过高）
)

