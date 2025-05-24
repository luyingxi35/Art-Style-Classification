import os
import json
import cv2
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm
import predict_utils
from predict_utils import process_image, load_checkpoint, predict
import torch

from PIL import Image
from PIL import ImageFile

# 加载配置
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

CSV_PATH      = cfg['csv_path']
IMG_DIR       = cfg['img_dir']
CHECKPOINT_PATH    = cfg['checkpoint_path']
OUTPUT_JSON   = 'train_data_for_shallow_2x9'
CNN_INPUT_SZ  = cfg['cnn_input_size']
OUTPUT_PATCH  = cfg['output_patches']
IN_TRAIN = True

STYLE_MAPPING = {
    'Art Nouveau (Modern)': ['Art Nouveau (Modern)'],
    'Baroque': ['Baroque'],
    'Expressionism': ['Expressionism'],
    'Impressionism': ['Impressionism'],
    'Post-Impressionism': ['Post-Impressionism'],
    'Rococo': ['Rococo'],
    'Romanticism': ['Romanticism'],
    'Surrealism': ['Surrealism'],
    'Symbolism': ['Symbolism']
}

LABEL_TO_INDEX = {
    'Art Nouveau (Modern)': 0,
    'Baroque': 1,
    'Expressionism': 2,
    'Impressionism': 3,
    'Post-Impressionism': 4,
    'Rococo': 5,
    'Romanticism': 6,
    'Surrealism': 7,
    'Symbolism': 8
}

NUM_CLASSES = len(LABEL_TO_INDEX)  # 类别总数

def label_to_one_hot(label, num_classes=NUM_CLASSES):
    one_hot = [0] * num_classes
    if label in LABEL_TO_INDEX:
        one_hot[LABEL_TO_INDEX[label]] = 1
    return one_hot

def map_style(style):
    for key, values in STYLE_MAPPING.items():
        if style in values:
            return key
    return 'Unknown'  # 如果风格不在映射中，返回 'Unknown'


def preprocess_and_extract_patches(pil_image, cnn_input_size, object_patch_path="../MainObjectSeek/object_patch/"):
    """
    返回两个patch:
    1. 原图resize为cnn_input_size×cnn_input_size
    2. object_patch同名文件resize为cnn_input_size×cnn_input_size（若不存在则用原图中心patch）
    """
    # 1. 原图resize
    patch1 = pil_image.resize((cnn_input_size, cnn_input_size), Image.BILINEAR)

    # 2. object_patch同名文件
    if object_patch_path and os.path.exists(object_patch_path):
        obj_img = Image.open(object_patch_path).convert('RGB')
        patch2 = obj_img.resize((cnn_input_size, cnn_input_size), Image.BILINEAR)
    else:
        # 用原图中心patch
        w, h = pil_image.size
        left = (w - cnn_input_size) // 2
        top = (h - cnn_input_size) // 2
        patch2 = pil_image.crop((left, top, left + cnn_input_size, top + cnn_input_size))
    return [patch1, patch2]



def predict_and_store_one(image_path, true_label_one_hot, model, cnn_input_size=CNN_INPUT_SZ,
                           output_dir=OUTPUT_PATCH, output_json_path=OUTPUT_JSON):
    os.makedirs(output_dir, exist_ok=True)
    img = Image.open(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    patches = preprocess_and_extract_patches(img, cnn_input_size)

    # 预测并收集分数
    scores = []
    for idx, patch in enumerate(patches):
        # x = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        probs, top_labels = predict(patch, model, 9)
        # 创建一个长度为 9 的列表，初始化为 0（因为有 9 个类别）
        prob_vector = [0.0] * 9

        # 根据 LABEL_TO_INDEX 的顺序填充概率值
        for label, prob in zip(top_labels, probs):
            index = LABEL_TO_INDEX[label]
            prob_vector[index] = prob

        scores.append(prob_vector)
        # 可选：保存 patch 本地查看
        # cv2.imwrite(os.path.join(output_dir, f"patch_{idx+1}.jpg"), patch)

    # 构建输出格式
    # result = {
    #     "input": os.path.basename(image_path),  # 图片文件名
    #     "scores": scores,                          # 五维数组
    #     "label": true_label_one_hot                # 标签
    # }
    result = {
        "input": scores,
        "label": true_label_one_hot,              # 标签
    }


    # 写入 JSON
    with open(output_json_path, 'w', encoding='utf-8') as jf:
        json.dump(result, jf, ensure_ascii=False, separators=(',', ':'))

    #print(f"结果已保存至 {output_json_path}")

if __name__ == '__main__':
    # 读取 CSV 文件
    df = pd.read_csv(CSV_PATH)
    if df.empty:
        raise ValueError("CSV 文件为空")
    # 加载模型（可选择在循环外加载以提高效率）
    model, _, _, _ = load_checkpoint(CHECKPOINT_PATH)

    os.makedirs(OUTPUT_JSON, exist_ok=True)

    # 初始化计数器字典
    style_count = {style: 0 for style in STYLE_MAPPING.keys()}
    max_per_class = 3000

    # 只处理测试集或训练集部分
    df = df[df['in_train'] == IN_TRAIN]

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Images"):
        img_path = None
        # 在 train_1 ~ train_9 文件夹中查找图片
        for i in range(1, 10):
            candidate = os.path.join(f'../raw_data/train_{i}', row['new_filename'])
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            continue

        # 映射风格
        true_lbl = map_style(row['style'])
        if true_lbl == 'Unknown':
            continue  # 跳过未知风格

        # 每种类别最多3000张
        if style_count[true_lbl] >= max_per_class:
            continue

        # 增加计数器
        style_count[true_lbl] += 1

        # 转换标签为 1-hot 向量
        true_lbl_one_hot = label_to_one_hot(true_lbl)

        # 定义输出 JSON 文件路径（以图片编号命名）
        output_json_path = os.path.join(OUTPUT_JSON, f"{row['new_filename'].split('.')[0]}.json")

        # 如果输出文件已存在则跳过
        if os.path.exists(output_json_path):
            continue

        # 调用预测函数并保存结果
        try:
            predict_and_store_one(
                image_path=img_path,
                true_label_one_hot=true_lbl_one_hot,
                model=model,
                output_json_path=output_json_path
            )
        except Exception as e:
            print(f"处理图片 {img_path} 时出错: {e}")