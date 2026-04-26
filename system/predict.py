import cv2
import numpy as np
from PIL import Image
import os

# 确保导入正确的DeeplabV3类
from deeplab import DeeplabV3


def process_single_image(input_path, output_path=None, count=False, name_classes=["background", "water"]):
    """
    处理单张图片并返回分割后的可视化结果

    参数:
        input_path: 输入图片的路径
        output_path: 输出图片的保存路径，为None时不保存
        count: 是否计算像素数量及比例
        name_classes: 类别名称列表

    返回:
        PIL.Image: 分割后的可视化图像
    """
    # 初始化模型
    deeplab = DeeplabV3()

    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入图片不存在: {input_path}")

    # 打开图片
    try:
        image = Image.open(input_path)
    except Exception as e:
        raise RuntimeError(f"无法打开图片: {str(e)}")

    # 进行分割处理
    r_image = deeplab.detect_image(image, count=count, name_classes=name_classes)

    # 如果指定了输出路径则保存图片
    if output_path:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        r_image.save(output_path)

    return r_image


if __name__ == "__main__":
    # 示例用法
    # 输入图片路径
    input_image_path = "cc2.jpg"
    # 输出图片路径，为None则不保存仅返回结果
    output_image_path = "output.jpg"

    try:
        # 处理图片
        result_image = process_single_image(
            input_path=input_image_path,
            output_path=output_image_path,
            count=True,  # 计算像素数量及比例
            name_classes=["background", "water"]  # 类别名称
        )
        # 显示结果（如果需要在后端环境显示）
        # result_image.show()
        print(f"处理完成，结果已保存至: {output_image_path}")
    except Exception as e:
        print(f"处理失败: {str(e)}")
