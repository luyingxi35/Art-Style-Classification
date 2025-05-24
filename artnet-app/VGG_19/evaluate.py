import os
import json
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
import torch.nn as nn
from sklearn.metrics import classification_report
import argparse
from PIL import Image
from PIL import ImageFile
import predict_utils
from predict_utils import process_image, load_checkpoint, predict
import torch

PATCH_NUM = 2
CLASS_NUM = 9
TEST_DATA_PATH = 'test_data_2x9'  # 测试数据集路径
SHALLOWNN_PATH = 'shallow_nn_2x9.pth'  # 模型路径
# TEST_DATA_PATH = 'test_data_for_shallow'  # 测试数据集路径
# TEST_DATA_PATH_16 = 'hierachy/test_data_for_shallow_16'  # 测试数据集路径
# TEST_DATA_PATH_4 = 'hierachy/test_data_for_shallow_4'  # 测试数据集路径
# TEST_DATA_PATH_5 = 'test_data_for_shallow'  # 测试数据集路径
IMAGE_PATH = '../testie'
BASELINE_CKPT_PATH = 'checkpoint5.pth'  # 模型路径
#SHALLOWNN_5_PATH = 'shallow_nn_5x5_29.pth'
# SHALLOWNN_5_PATH = 'shallow_nn_5x5.pth'
# SHALLOWNN_16_PATH = 'shallow_nn_16x16.pth'
# SHALLOWNN_4_PATH = 'shallow_nn_4x4.pth'
IS_BASELINE = False  # 是否使用基线模型

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='设置模型 ID')

# 添加 --model_id 参数，类型为整数，默认值为 2
parser.add_argument('--id', type=int, default=0, help='模型 ID')

# 解析命令行参数
args = parser.parse_args()

# 使用解析后的 model_id
MODEL_ID = args.id

# MODEL_ID:
# shallow_nn_5x5: 0
# shallow_nn_4x4: 1
# shallow_nn_16x16: 2

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


class ShallowNN(nn.Module):
    def __init__(self, input_dim=PATCH_NUM*CLASS_NUM, hidden_dim=128, hidden_dim1=64, hidden_dim2=32, output_dim=CLASS_NUM, dropout_rate=0.3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim1)
        self.bn2 = nn.BatchNorm1d(hidden_dim1)
        self.fc3 = nn.Linear(hidden_dim1, output_dim)
        self.bn3 = nn.BatchNorm1d(output_dim)
   
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        return x
    
class ShallowNN_2(nn.Module):
    def __init__(self, input_dim=80, hidden_dim1=128, hidden_dim2=64, hidden_dim3=32, output_dim=5):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim3, output_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, folder_path):
        self.data = []
        # 遍历文件夹中的所有 JSON 文件
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.json'):  # 只处理 JSON 文件
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'r') as f:
                    file_data = json.load(f)
                    self.data.append(file_data)
        
        # 输入为 5×5 的张量
        self.inputs = [item['input'] for item in self.data]
        # 输出为单个 5 维向量
        self.labels = [torch.tensor(item['label'], dtype=torch.float32) for item in self.data]
        self.scores = [torch.tensor(item['scores'], dtype=torch.float32) for item in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx], self.scores[idx]

def generate_pridiction_for_baseline(model, image_path):
    image = Image.open(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # image = tf.convert_to_tensor(image, dtype=tf.float32)
    # image = tf.expand_dims(image, axis=0)  # 替代 np.expand_dims
    probs, top_labels = predict(image, model, 9)
    # 创建一个长度为 9 的列表，初始化为 0（因为有 9 个类别）
    prob_vector = [0.0] * CLASS_NUM

    # 根据 LABEL_TO_INDEX 的顺序填充概率值
    for label, prob in zip(top_labels, probs):
        index = LABEL_TO_INDEX[label]
        prob_vector[index] = prob


    label = np.argmax(np.array(prob_vector))

    return label

def generate_pridiction_for_shallownn(model, scores):
    with torch.no_grad():  # 不计算梯度，推理模式
        scores = scores.unsqueeze(0)  # 加一维，变成 batch_size=1
        pred = model(scores)
    label = np.argmax(np.array(pred))
    return label

def compute_correction_rate(model, test_loader):
    correct_count = 0
    total_count = 0
    style_total = np.zeros(9)
    style_correct = np.zeros(9)

    all_preds = []
    all_labels = []

    for inputs, labels, scorces in test_loader:
        for idx in range(len(inputs)):
            total_count += 1
            label_one_hot = labels[idx]
            label = np.argmax(np.array(label_one_hot))
            style_total[label] += 1

            if IS_BASELINE:
                input_path = inputs[idx]
                image_path = os.path.join(IMAGE_PATH, input_path)
                print(f"Processing {image_path}({total_count}/500)...")
                predicted_label = generate_pridiction_for_baseline(model, image_path)
            else:
                scores = scorces[idx]
                print(f"Processing {total_count}/500...")
                predicted_label = generate_pridiction_for_shallownn(model, scores)

            all_preds.append(predicted_label)
            all_labels.append(label)

            if predicted_label == label:
                correct_count += 1
                style_correct[label] += 1

    style_correction_rate = style_correct / style_total

    return correct_count / total_count,  style_correction_rate, style_total, style_correct, all_preds, all_labels

if __name__ == '__main__':
    # if MODEL_ID == 0:
    #     TEST_DATA_PATH = TEST_DATA_PATH_5
    #     SHALLOWNN_PATH = SHALLOWNN_5_PATH
    # elif MODEL_ID == 1:
    #     TEST_DATA_PATH = TEST_DATA_PATH_4
    #     SHALLOWNN_PATH = SHALLOWNN_4_PATH
    # elif MODEL_ID == 2:
    #     TEST_DATA_PATH = TEST_DATA_PATH_16
    #     SHALLOWNN_PATH = SHALLOWNN_16_PATH
    # else:
    #     raise ValueError("Invalid MODEL_ID. Must be 0, 1, or 2.")
    test_dataset = CustomDataset(TEST_DATA_PATH)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

    if IS_BASELINE:
        model, _, _, _ = load_checkpoint(BASELINE_CKPT_PATH)
    else:
        model = ShallowNN(input_dim=PATCH_NUM*CLASS_NUM)
        model.load_state_dict(torch.load(SHALLOWNN_PATH))
        model.eval()

    accuracy, style_accuracy, style_total, style_correct, all_preds, all_labels = compute_correction_rate(model, test_loader)

    print(f"\nTotal Accuracy: {accuracy * 100:.2f}%\n")

    labels = ['Art Nouveau (Modern)', 'Baroque', 'Expressionism', 'Impressionism', 'Post-Impressionism', 'Rococo', 'Romanticism',
         'Surrealism', 'Symbolism']
    for i, name in enumerate(labels):
        print(f"{name} Accuracy: {style_accuracy[i] * 100:.2f}% | Total: {style_total[i]} | Correct: {style_correct[i]}")

    print("\nClassification Report (Precision / Recall / F1-Score):")
    print(classification_report(
    all_labels, all_preds,
    labels=list(range(len(labels))),
    target_names=labels,
    digits=4,
    zero_division=0
))  

#6-patch
#                       precision    recall  f1-score   support

# Art Nouveau (Modern)     0.0000    0.0000    0.0000         0
#              Baroque     0.6364    0.0700    0.1261       100
#        Expressionism     0.3158    0.4200    0.3605       100
#        Impressionism     0.3231    0.7119    0.4444        59
#   Post-Impressionism     0.2683    0.2683    0.2683        41
#               Rococo     0.0000    0.0000    0.0000         0
#          Romanticism     0.0000    0.0000    0.0000         0
#           Surrealism     0.4352    0.4700    0.4519       100
#            Symbolism     0.0000    0.0000    0.0000       100

#             accuracy                         0.2980       500
#            macro avg     0.2199    0.2156    0.1835       500
#         weighted avg     0.3376    0.2980    0.2622       500
