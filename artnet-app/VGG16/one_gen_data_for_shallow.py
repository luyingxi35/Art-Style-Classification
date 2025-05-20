import os
import cv2
import yaml
import pandas as pd
from tensorflow.keras.models import load_model
from tqdm import tqdm
from two_patch_json_generator import JSONGenerator

# 加载配置
with open('one_config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

CSV_PATH      = cfg['csv_path']
IMG_DIR      = cfg['img_dir']  # 这是一个列表
MODEL_PATH    = cfg['model_path']
OUTPUT_JSON   = cfg['output_json']
OUTPUT_IMG    = cfg['output_img']
CNN_INPUT_SZ  = cfg['cnn_input_size']
INTRAIN      = cfg['in_train']

STYLES = [
            'Minimalism', 'Romanticism', 'Rococo', 'Post_Impressionism', 'Art_Nouveau_Modern',
            'Renaissance', 'Pointillism', 'Realism', 'Ukiyo_e', 'Symbolism', 'Baroque', 'Cubism',
            'Abstract', 'Pop_Art', 'Impressionism', 'Expressionism', 'Color_Field_Painting'
        ]

def find_image_path(img_dir, filename):
    if not isinstance(img_dir, list):
        img_dir = [img_dir]
    for folder in img_dir:
        img_path = os.path.join(folder, filename)
        if os.path.exists(img_path):
            return img_path
    return None

# 读取 CSV 文件
df = pd.read_csv(CSV_PATH)
if df.empty:
    raise ValueError("CSV 文件为空")
# 加载模型（可选择在循环外加载以提高效率）
model = load_model(MODEL_PATH)

# 确保输出 JSON 文件的目录存在
os.makedirs(OUTPUT_JSON, exist_ok=True)
os.makedirs(OUTPUT_IMG, exist_ok=True)
# 遍历每一行，生成对应的 JSON 文件
# 筛选测试集部分
df = df[df['in_train'] == INTRAIN]  # 筛选出测试/训练集部分


style_count = {style: 0 for style in STYLES}
max_per_style = 1000

json_gen = JSONGenerator(
    model_path=MODEL_PATH,
    img_size=CNN_INPUT_SZ   
)


# 遍历每一行，生成对应的 JSON 文件
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Images"):
    img_path = find_image_path(IMG_DIR, row['new_filename'])
    if img_path is None:
        continue

    # 映射风格
    true_lbl = row['style']
    if true_lbl not in STYLES:
        continue

    # 检查该风格是否已达到 20 张
    if style_count[true_lbl] >= max_per_style:
        continue

    # 增加计数器
    style_count[true_lbl] += 1

    # 把该图片存到 testie 文件夹中
    cv2.imwrite(os.path.join(OUTPUT_IMG, row['new_filename']), cv2.imread(img_path))

    # 定义输出 JSON 文件路径（以图片编号命名）
    output_json_path = os.path.join(OUTPUT_JSON, f"{row['new_filename'].split('.')[0]}.json")

    # 调用预测函数并保存结果
    try:
        json_gen.generate_json(
            image_path=img_path,
            output_json=output_json_path,
            label=true_lbl
        )
    except Exception as e:
        print(f"处理图片 {img_path} 时出错: {e}")

# 输出每种风格的图片数量
print("每种风格的图片数量：")   
for style, count in style_count.items():
    print(f"{style}: {count} 张")
# 输出总图片数量
print(f"总图片数量: {sum(style_count.values())} 张")

# 每种风格的图片数量：
# Minimalism: 314 张
# Romanticism: 1000 张
# Rococo: 1000 张
# Post_Impressionism: 0 张
# Art_Nouveau_Modern: 0 张
# Renaissance: 0 张
# Pointillism: 387 张
# Realism: 1000 张
# Ukiyo_e: 0 张
# Symbolism: 1000 张
# Baroque: 1000 张
# Cubism: 1000 张
# Abstract: 0 张
# Pop_Art: 0 张
# Impressionism: 1000 张
# Expressionism: 1000 张
# Color_Field_Painting: 0 张
# 总图片数量: 8701 张