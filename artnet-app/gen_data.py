import os
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# 假设你已经实现了dividing_patches方法
def preprocess_and_extract_patches(image_path, cnn_input_size=325, output_dir="patches"):
    """
    预处理图像并提取五个固定位置的图像块，并将它们保存到指定目录。
    
    参数：
        image_path (str): 输入图像的路径。
        cnn_input_size (int): CNN模型的输入尺寸（默认为325）。
        output_dir (str): 保存图像块的目录（默认为"patches"）。
    
    返回：
        patches (list): 包含五个图像块的列表。
    """
    os.makedirs(output_dir, exist_ok=True)
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("图像读取失败，请检查图像路径。")
    
    resized_image = cv2.resize(image, (2 * cnn_input_size, 2 * cnn_input_size))
    patches = []
    
    patch_size = cnn_input_size
    p1 = resized_image[:patch_size, :patch_size]
    p2 = resized_image[:patch_size, -patch_size:]
    p3 = resized_image[-patch_size:, :patch_size]
    p4 = resized_image[-patch_size:, -patch_size:]
    center_y = resized_image.shape[0] // 2
    center_x = resized_image.shape[1] // 2
    p5 = resized_image[center_y - patch_size//2 : center_y + patch_size//2,
                       center_x - patch_size//2 : center_x + patch_size//2]
    
    patches = [p1, p2, p3, p4, p5]
    
    for i, patch in enumerate(patches):
        output_path = os.path.join(output_dir, f"patch_{i+1}.jpg")
        cv2.imwrite(output_path, patch)
    
    return patches

def predict_and_store(image_path, true_label, model, cnn_input_size=325, output_dir="patches", output_json_path="results.json"):
    """
    对图像进行patch划分，预测每个patch的风格，并将结果存储为JSON格式。
    
    参数：
        image_path (str): 输入图像的路径。
        true_label (str): 图像的真实风格标签。
        model: 预训练的风格分类模型。
        cnn_input_size (int): CNN模型的输入尺寸（默认为325）。
        output_dir (str): 保存图像块的目录（默认为"patches"）。
        output_json_path (str): 保存结果的JSON文件路径。
    
    返回：
        None
    """
    # 提取patches
    patches = preprocess_and_extract_patches(image_path, cnn_input_size, output_dir)
    
    # 存储结果
    results = {
        "input": [],
        "label": true_label
    }
    
    for i, patch in enumerate(patches):
        # 转换图像格式
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        patch = patch.astype(np.float32) / 255.0
        
        # 预测
        pred_score = model.predict(np.expand_dims(patch, axis=0))[0]
        
        # 保存结果
        results["input"].append(pred_score.tolist())
    
    # 将结果存储为JSON文件
    with open(output_json_path, "w") as json_file:
        json.dump(results, json_file)
    
    print(f"结果已保存到 {output_json_path}")

# 示例使用
if __name__ == "__main__":
    # 加载模型
    model_path = "model/artnet"  # 替换为你的模型路径
    model = load_model(model_path)
    
    # 图像路径和真实标签
    image_path = "sample_images/munch.jpg"  # 替换为你的图像路径
    true_label = "Impressionism"  # 替换为图像的真实风格标签
    
    # 输出目录和JSON文件路径
    output_dir = "data"
    output_json_path = "data_for_shallow.json"
    
    try:
        predict_and_store(image_path, true_label, model, output_dir=output_dir, output_json_path=output_json_path)
    except Exception as e:
        print(f"发生错误：{e}")