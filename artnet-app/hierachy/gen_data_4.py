import os
import json
import cv2
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

# 加载配置
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

CSV_PATH      = cfg['csv_path']
IMG_DIR       = cfg['img_dir']
MODEL_PATH    = cfg['model_path']
OUTPUT_JSON   = cfg['output_json']
CNN_INPUT_SZ  = cfg['cnn_input_size']
OUTPUT_PATCH  = cfg['output_patches']

STYLE_MAPPING = {
    'Cubism': ['Cubism', 'Tubism', 'Cubo-Expressionism', 'Mechanistic Cubism', 'Analytical Cubism', 'Cubo-Futurism', 'Synthetic Cubism'],
    'Impressionism': ['Impressionism', 'Post-Impressionism', 'Synthetism', 'Divisionism', 'Cloisonnism'],
    'Expressionism': ['Expressionism', 'Neo-Expressionism', 'Figurative Expressionism', 'Fauvism'],
    'Realism': ['Realism', 'Hyper-Realism', 'Photorealism', 'Analytical Realism', 'Naturalism'],
    'Abstract': ['Abstract Art', 'New Casualism', 'Post-Minimalism', 'Orphism', 'Constructivism', 'Lettrism', 'Neo-Concretism', 'Suprematism',
                 'Spatialism', 'Conceptual Art', 'Tachisme', 'Post-Painterly Abstraction', 'Neoplasticism', 'Precisionism', 'Hard Edge Painting']
}

LABEL_TO_INDEX = {
    'Cubism': 0,
    'Expressionism': 1,
    'Impressionism': 2,
    'Realism': 3,
    'Abstract': 4
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


def preprocess_and_extract_patches(image, cnn_input_size=CNN_INPUT_SZ):
    target = 2 * cnn_input_size
    h, w = image.shape[:2]
    scale = target / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    pad_w = target - new_w
    pad_h = target - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=[0,0,0])
    p = cnn_input_size
    patches = [
        padded[0:p,       0:p      ],
        padded[0:p,       -p:      ],
        padded[-p:,       0:p      ],
        padded[-p:,       -p:      ]
    ]
    return patches


def apply_data_augmentation(image):
    """
    对图像数据应用多种不同的增强方式，生成多个增强后的图像
    :param image: 输入图像
    :return: 增强后的图像列表
    """
    datagen = ImageDataGenerator(
        rotation_range=30,          # 随机旋转角度范围
        width_shift_range=0.3,      # 宽度偏移范围
        height_shift_range=0.3,     # 高度偏移范围
        shear_range=0.3,            # 剪切强度
        zoom_range=0.3,             # 随机缩放范围
        horizontal_flip=True,       # 随机水平翻转
        brightness_range=[0.3, 1.7] # 亮度变化范围
    )

    # 将图像转换为批次格式
    x = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    # 生成多个增强后的图像
    augmented_images = []
    for _ in range(5):  # 假设生成5个增强版本
        augmented_x = datagen.random_transform(x[0])
        augmented_images.append((augmented_x * 255).astype(np.uint8))

    return augmented_images


def predict_and_store_augmented(image_path, true_label_one_hot, model, cnn_input_size=CNN_INPUT_SZ,
                                output_dir=OUTPUT_PATCH, output_json_dir=OUTPUT_JSON):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_json_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 应用数据增强，生成多个版本
    augmented_images = apply_data_augmentation(img)

    # 预测并为每个增强版本生成独立的 JSON 文件
    for idx, aug_img in enumerate(augmented_images):
        patches = preprocess_and_extract_patches(aug_img, cnn_input_size)

        scores = []
        for patch in patches:
            x = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            pred = model.predict(np.expand_dims(x, axis=0))[0]
            scores.append(pred.tolist())

        # 构建输出格式
        result = {
            "input": f"{os.path.basename(image_path).split('.')[0]}_aug_{idx+1}",  # 增强图片的唯一标识
            "scores": scores,                          # 五维数组
            "label": true_label_one_hot                # 标签
        }

        # 定义输出 JSON 文件路径
        output_json_path = os.path.join(output_json_dir, f"{os.path.basename(image_path).split('.')[0]}_aug_{idx+1}.json")

        # 写入 JSON
        with open(output_json_path, 'w', encoding='utf-8') as jf:
            json.dump(result, jf, ensure_ascii=False, separators=(',', ':'))

        print(f"结果已保存至 {output_json_path}")


if __name__ == '__main__':
    # 读取 CSV 文件
    df = pd.read_csv(CSV_PATH)
    if df.empty:
        raise ValueError("CSV 文件为空")
    # 加载模型（可选择在循环外加载以提高效率）
    model = load_model(MODEL_PATH)

    # 确保输出 JSON 文件的目录存在
    os.makedirs(OUTPUT_JSON, exist_ok=True)
    os.makedirs("testie", exist_ok=True)

    # 遍历每一行，生成对应的 JSON 文件（带数据增强）
    # 筛选测试集部分
    df = df[df['in_train'] == False]  # 筛选出测试集部分
    if len(df) < 500:
        raise ValueError("测试集数据不足 500 行")
    
    # 初始化计数器字典
    style_count = {style: 0 for style in STYLE_MAPPING.keys()}

    # 遍历每一行，生成对应的 JSON 文件
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Images"):
        img_path = os.path.join(IMG_DIR, row['new_filename'])  # 获取图片路径

        # 跳过不存在的图片
        if not os.path.exists(img_path):
            continue

        # 映射风格
        true_lbl = map_style(row['style'])
        if true_lbl == 'Unknown':
            continue  # 跳过未知风格

        # 检查该风格是否已达到 20 张
        if style_count[true_lbl] >= 100:
            continue

        # 增加计数器
        style_count[true_lbl] += 1

        # 把该图片存到 testie 文件夹中
        cv2.imwrite(os.path.join("testie", row['new_filename']), cv2.imread(img_path))

        # 转换标签为 1-hot 向量
        true_lbl_one_hot = label_to_one_hot(true_lbl)

        # 调用预测函数并保存结果
        try:
            predict_and_store_augmented(
                image_path=img_path,
                true_label_one_hot=true_lbl_one_hot,
                model=model
            )
        except Exception as e:
            print(f"处理图片 {img_path} 时出错: {e}")