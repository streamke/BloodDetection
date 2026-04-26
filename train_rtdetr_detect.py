import os

# 解决显存碎片问题（必须放在代码最开头）
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import gc
from ultralytics import RTDETR
import warnings

# 忽略无关警告
warnings.filterwarnings('ignore')


# 清理GPU显存
def clean_gpu_memory():
    torch.cuda.empty_cache()  # 清空PyTorch缓存显存
    gc.collect()  # 回收Python垃圾内存
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated(0) / (1024 ** 3)  # 已用显存（GB）
        free = torch.cuda.memory_reserved(0) / (1024 ** 3) - used  # 空闲显存（GB）
        print(f"✅ GPU显存状态：已使用{used:.2f}GB，空闲{free:.2f}GB")


if __name__ == '__main__':
    # 1. 校验模型配置文件路径
    model_cfg_path = r"C:\Users\25896\Desktop\ultralytics-main\ultralytics-main\ultralytics\cfg\models\rt-detr\test.yaml"
    if not os.path.exists(model_cfg_path):
        raise FileNotFoundError(f"❌ 模型配置文件不存在！路径：{model_cfg_path}")

    # 2. 清理显存并加载RT-DETR模型
    clean_gpu_memory()
    model = RTDETR(model_cfg_path)

    # 加载预训练权重（确保rtdetr-l.pt路径正确）
    rtdetr_weight_path = r"C:\Users\25896\Desktop\ultralytics-main\ultralytics-main\ultralytics\RT-DETR-Car\exp_final5\weights\best.pt"
    if os.path.exists(rtdetr_weight_path):
        model.load(rtdetr_weight_path)
        print(f"✅ 成功加载RT-DETR权重：{rtdetr_weight_path}")
    else:
        raise FileNotFoundError(f"❌ RT-DETR权重文件不存在！路径：{rtdetr_weight_path}")

    # 3. 校验数据集配置文件路径
    data_cfg_path = r'C:\Users\25896\Desktop\ultralytics-main\ultralytics-main\data\che.yaml'
    if not os.path.exists(data_cfg_path):
        raise FileNotFoundError(f"❌ 数据集配置文件不存在！路径：{data_cfg_path}")

    # 4. 启动训练（移除accumulate参数，适配旧版本Ultralytics）
    print("🚀 开始训练...")
    model.train(
        data=data_cfg_path,
        epochs=150,
        patience=15,  # 早停机制，避免无效训练
        batch=4,  # 8GB显存安全Batch Size（无accumulate时优先保证不溢出）
        imgsz=640,  # 降低图像尺寸，减少显存占用
        amp=True,  # 启用AMP（需确保yolo11n.pt已正确放置）
        cache=True,  # 缓存数据集，减少磁盘IO
        workers=0,  # 减少线程，避免CPU内存占用过高
        plots=False,  # 关闭可视化图表，节省显存
        mixup=0.0,  # 临时关闭mixup，进一步降低显存消耗
        mosaic=1.0,  # 保留mosaic增强（显存压力小，不影响训练）
        device=0,  # 使用第0块GPU
        # optimizer='AdamW',  # 优化器
        # lr0=1e-4,  # 初始学习率（适配小Batch）
        # lrf=0.01,  # 学习率衰减因子
        # cos_lr=True,  # 余弦学习率调度，收敛更稳定
        project='RT-DETR-Car',  # 训练结果保存项目名
        name='exp_final'  # 实验名，避免覆盖之前结果
    )
    print("🎉 训练完成！结果已保存至：RT-DETR-Car/exp_final")