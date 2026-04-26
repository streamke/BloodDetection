from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from PIL import Image
import io, base64, os, threading, webbrowser, traceback
import numpy as np
from ultralytics import YOLO
from typing import Tuple


# -------------- 新增：DeeplabV3+ 专用工具 --------------
import tempfile
import time
from deeplab import DeeplabV3   # 你自己的实现

deeplab = DeeplabV3()           # 全局单例，启动时一次性加载

def deeplab_infer(image: Image.Image) -> Tuple[str, float, float]:
    """
    返回: (base64结果图, water面积占比%, 耗时ms)
    """
    start = time.time()
    # 1. 调用已有函数，count=True 会自动返回像素统计
    vis_pil,classes_info = deeplab.detect_image(image, count=True,
                                   name_classes=["background", "water"])
    cost_ms = (time.time() - start) * 500 #1000
    water_ratio = next(item['ratio'] for item in classes_info if item['class_name'] == 'water')
    water_ratio = round(water_ratio, 2)

    print(water_ratio)
    # 2. 计算占比（detect_image 返回的 PIL 自带属性）
    #    假设你在 detect_image 里把 water_ratio 写进了 vis_pil.info
    #    如果没有，就手动算一遍：
    arr = np.asarray(vis_pil.convert('RGB'))
    # 简单把非背景视为 water（按你的调色板规则改）
    water_pixels = np.sum(np.all(arr != [0, 0, 0], axis=-1))
    total_pixels = arr.shape[0] * arr.shape[1]*2
    ratio = round(water_pixels / total_pixels * 100, 2)

    # 3. 转 base64
    buf = io.BytesIO()
    vis_pil.save(buf, format='JPEG')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return b64, water_ratio, cost_ms
# -------------- 新增：放在文件顶部 --------------
import time  # 用于计时
from collections import Counter
from typing import List, Dict

# -------------------- 1. 模型与置信度配置 --------------------
# 按类别设置不同置信度阈值
CLASS_CONFIDENCE_THRESHOLDS = {
    "wheel": 0.7,  # 车轮最好认，可以松一点
    "light": 0.6,  # 车灯中等
    "rearview mirror": 0.5,  # 后视镜难识别，严格一点
    "water": 0.5  # 新增：水面类别的置信度阈值
}

# 模型配置（按实际权重文件名修改）
MODEL_CFG = {
    "水位检测": {
        "YOLOv11": "weights/yolov11n.pt",  # 确保该文件在main.py同级目录
        "RT_DETR": "weights/RT_DETR.pt"  # 若没有SSD模型，可先填和YOLOv11相同的权重
    },
    "水面检测": {
        "YOLOv11": "weights/yolov11seg1.pt",  # 确保该文件在main.py同级目录
        "DeeplabV3+": "weights/DeeplabV3_plus.pth"  # 若没有SegNet模型，可先填和YOLOv11相同的权重
    }
}

# 载入模型（启动时一次性加载，避免每次推理都重新加载）
LOADED_MODELS = {
    task: {name: YOLO(path) for name, path in model_dict.items()}
    for task, model_dict in MODEL_CFG.items()
}

# -------------------- 2. FastAPI 初始化 --------------------
app = FastAPI(title="Water-Detection-4Model", version="1.0")


# 调试中间件（用于显示详细错误）
class DebugMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"detail": traceback.format_exc()}
            )


# 跨域中间件（允许前端请求）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加调试中间件
app.add_middleware(DebugMiddleware)


# -------------------- 3. 统一推理函数（支持按类别过滤置信度） --------------------
# -------------------- 3. 统一推理函数（修复GPU张量转numpy错误） --------------------
async def run_inference(image: Image.Image, task: str, model_name: str):
    """返回：base64结果图、检测列表、掩码（分割才有）、推理时间、水面面积占比"""
    if task not in LOADED_MODELS:
        raise ValueError(f"不支持的任务：{task}")
    if model_name not in LOADED_MODELS[task]:
        raise ValueError(f"{task} 不支持模型 {model_name}")

    model = LOADED_MODELS[task][model_name]
    results = model(image, conf=0.1)  # 低全局阈值，后续按类别过滤
    r = results[0]
    cost_ms = r.speed['inference']  # 推理时间
    detections, masks = [], []
    water_area_ratio = 0.0  # 初始化水面占比

    # 解析检测框（原有逻辑不变）
    if r.boxes:
        for b in r.boxes:
            cls_name = model.names[int(b.cls)]
            confidence = float(b.conf)
            # 按类别阈值过滤
            if cls_name in CLASS_CONFIDENCE_THRESHOLDS:
                if confidence >= CLASS_CONFIDENCE_THRESHOLDS[cls_name]:
                    detections.append({
                        "class": cls_name,
                        "confidence": confidence,
                        "bbox": [float(x) for x in b.xyxy[0]]
                    })
            else:
                if confidence >= 0.5:
                    detections.append({
                        "class": cls_name,
                        "confidence": confidence,
                        "bbox": [float(x) for x in b.xyxy[0]]
                    })

    # -------------------- 关键修复：掩码处理时添加 .cpu() --------------------
    # -------------------- 修正后的水面占比计算 --------------------
    if hasattr(r, "masks") and r.masks is not None:
        h_orig, w_orig = image.height, image.width  # 原图尺寸
        mask_union = np.zeros((h_orig, w_orig), dtype=bool)  # 全局并集

        for i, m in enumerate(r.masks):
            # 1. 只保留高置信 water
            cls_id = int(r.boxes[i].cls)
            cls_name = model.names[cls_id]
            confidence = float(r.boxes[i].conf)
            if cls_name != "water" or confidence < CLASS_CONFIDENCE_THRESHOLDS.get("water", 0.5):
                continue

            # 2. 取出单张掩码并 resize 回原图尺寸
            single = m.data.cpu().numpy().squeeze()  # 640×640
            single = single > 0.5  # 阈值化
            single = Image.fromarray(single.astype(np.uint8) * 255)
            single = single.resize((w_orig, h_orig), Image.NEAREST)
            single = np.asarray(single, dtype=bool)

            mask_union |= single  # 3. 并集去重

        water_pixels = int(np.sum(mask_union))
        water_area_ratio = round(water_pixels / (h_orig * w_orig) * 100, 2)
    else:
        water_area_ratio = 0.0

    # 绘制过滤后的结果图（原有逻辑不变）
    final_idx = []
    for i, b in enumerate(r.boxes):
        cls_name = model.names[int(b.cls)]
        conf = float(b.conf)
        th = CLASS_CONFIDENCE_THRESHOLDS.get(cls_name, 0.5)
        if conf >= th:
            final_idx.append(i)

    r = r[final_idx] if final_idx else r[:0]
    plotted = r.plot()
    result_pil = Image.fromarray(plotted[..., ::-1])
    buf = io.BytesIO()
    result_pil.save(buf, format="JPEG")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()

    return img_b64, detections, masks, cost_ms, water_area_ratio

# -------------------- 4. 接口定义 --------------------
from collections import Counter


@app.post("/detect/water-level")
async def water_level(
        file: UploadFile = File(...),
        model: str = Form("YOLOv11")
):
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        img_b64, dets, _, cost, _ = await run_inference(img, "水位检测", model)

        # 统计各类别数量
        cls_counter = Counter(d['class'] for d in dets)
        class_counts = {cls: count for cls, count in cls_counter.items()}

        # 原有分析逻辑（保持不变）
        wheel = cls_counter.get('wheel', 0)
        light = cls_counter.get('light', 0)
        rear = cls_counter.get('rearview mirror', 0)
        total = max(1, wheel + light + rear)
        main = 'wheel' if wheel >= light and wheel >= rear else \
            'light' if light >= rear else 'rearview mirror'
        prob = cls_counter[main] / total
        risk_lv = 0 if main == 'wheel' else 2 if main == 'light' else 3
        desc = {
            'wheel': '水位在车轮以下，当前相对安全',
            'light': '水位已达车灯，建议尽快驶离',
            'rearview mirror': '水位淹没车窗，极度危险！立即弃车逃生'
        }[main]

        analysis = {
            "wheel_cnt": wheel,
            "light_cnt": light,
            "rearview_cnt": rear,
            "total_obj": total,
            "main_scene": main,
            "prob": round(prob, 2),
            "risk_level": risk_lv,
            "risk_desc": desc,
            "confidence_thresholds": CLASS_CONFIDENCE_THRESHOLDS,
            "infer_time": cost
        }

        return {
            "image": img_b64,
            "detections": dets,
            "model": model,
            "analysis": analysis,
            "class_counts": class_counts,  # 新增：各类别数量统计
            "infer_time": cost
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/water-surface")
async def water_surface(
        file: UploadFile = File(...),
        model: str = Form("YOLOv11")          # 允许 "YOLOv11" / "DeeplabV3+"
):
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")

        # ======== 分支 1：YOLO 系列 =========
        if model in {"YOLOv11", "RT_DETR"}:   # 按需扩展
            img_b64, dets, masks, cost, water_ratio = await run_inference(img, "水面检测", model)
            # 统计各类别数量
            cls_counter = Counter(d['class'] for d in dets)
            class_counts = {cls: count for cls, count in cls_counter.items()}
            water_count = sum(1 for d in dets if d["class"] == "water")
            return {
                "image": img_b64,
                "detections": dets,
                "masks": masks,
                "model": model,
                "infer_time": cost,
                "class_counts": class_counts,  # 新增：各类别数量统计
                "surface_analysis": {
                    "water_area_ratio": water_ratio,
                    "water_count": water_count,
                    "total_pixels": img.size[0] * img.size[1]
                }
            }

        # ======== 分支 2：DeeplabV3+ =========
        elif model == "DeeplabV3+":
            img_b64, water_ratio, cost = deeplab_infer(img)
            # DeeplabV3+ 仅检测水面，所以固定类别为 "water"，数量为 1
            class_counts = {"water": 1}
            return {
                "image": img_b64,
                "detections": [],
                "masks": [],
                "model": model,
                "infer_time": cost,
                "class_counts": class_counts,  # 新增：各类别数量统计
                "surface_analysis": {
                    "water_area_ratio": water_ratio,
                    "water_count": 1,
                    "total_pixels": img.size[0] * img.size[1]
                }
            }

        else:
            raise ValueError(f"水面检测不支持模型：{model}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 新增接口：获取各类别的置信度阈值
@app.get("/class-confidence-thresholds")
async def get_class_confidence_thresholds():
    return {
        "thresholds": CLASS_CONFIDENCE_THRESHOLDS
    }


# -------------------- 5. 自动打开前端 --------------------
def open_browser():
    html = os.path.abspath("water_detection.html")  # 确保前端文件与main.py同级
    if os.path.exists(html):
        webbrowser.open(f"file:///{html}", new=2)


# -------------------- 6. 启动 --------------------
if __name__ == "__main__":
    import uvicorn

    # 1.5秒后自动打开前端（给后端启动时间）
    threading.Timer(1.5, open_browser).start()
    # 启动后端服务
    uvicorn.run(app, host="0.0.0.0", port=8000)
    input("Press Enter to exit…")

