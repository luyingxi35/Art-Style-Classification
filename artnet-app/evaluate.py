import os
import json
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
import torch.nn as nn


TEST_DATA_PATH = 'test_data_for_shallow'  # 测试数据集路径
IMAGE_PATH = 'testie'
BASELINE_MODEL_PATH = 'model/artnet'  # 模型路径
SHALLOWNN_PATH = 'shallow_nn_5x5.pth'
IS_BASELINE = False  # 是否使用基线模型

LABEL_TO_INDEX = {
    'Cubism': 0,
    'Expressionism': 1,
    'Impressionism': 2,
    'Realism': 3,
    'Abstract': 4
}

class ShallowNN(nn.Module):
    def __init__(self, input_dim=25, hidden_dim=128, output_dim=5):
        super().__init__()
        self.flatten = nn.Flatten()  # 将5×5输入展平为25维
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.flatten(x)  # 展平操作：形状从 (batch,5,5) 变为 (batch,25)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
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
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = tf.convert_to_tensor(image)  # 转换为 Tensor
    pred = model.predict(np.expand_dims(image, axis=0))[0]

    label = np.argmax(np.array(pred))

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
    style_total = np.zeros(5)
    style_correct = np.zeros(5)

    for inputs, labels, scorces in test_loader:
        for idx in range(len(inputs)):
            total_count += 1
            label_one_hot = labels[idx]
            label = np.argmax(np.array(label_one_hot))
            style_total[label] += 1
            if IS_BASELINE:
                input_path = inputs[idx]
                image_path = os.path.join(IMAGE_PATH, input_path)
                print(f"Processing {image_path}({total_count}/100)...")
                predicted_label = generate_pridiction_for_baseline(model, image_path)
                if predicted_label == label:
                    correct_count += 1
                    style_correct[label] += 1
            else:
                scores = scorces[idx]
                print(f"Processing {total_count}/100...")
                predicted_label = generate_pridiction_for_shallownn(model, scores)
                if predicted_label == label:
                    correct_count += 1
                    style_correct[label] += 1

    style_correction_rate = style_correct / style_total

    return correct_count / total_count,  style_correction_rate, style_total, style_correct

if __name__ == '__main__':
    test_dataset = CustomDataset(TEST_DATA_PATH)  # 使用文件夹路径
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    if IS_BASELINE:
        model = load_model(BASELINE_MODEL_PATH)
    else:
        model = ShallowNN()
        model.load_state_dict(torch.load(SHALLOWNN_PATH))
        model.eval() 
    accuracy, style_accuract, style_total, style_currect = compute_correction_rate(model, test_loader)
    print(f"Total Accuracy: {accuracy * 100:.2f}%")
    print(f"Cubism Accuracy: {style_accuract[0] * 100:.2f}%")
    print(f"Cubism Total: {style_total[0]}")
    print(f"Cubism Correct: {style_currect[0]}")
    print(f"Expressionism Accuracy: {style_accuract[1] * 100:.2f}%")
    print(f"Expressionism Total: {style_total[1]}")
    print(f"Expressionism Correct: {style_currect[1]}")
    print(f"Impressionism Accuracy: {style_accuract[2] * 100:.2f}%")
    print(f"Impressionism Total: {style_total[2]}")
    print(f"Impressionism Correct: {style_currect[2]}")
    print(f"Realism Accuracy: {style_accuract[3] * 100:.2f}%")
    print(f"Realism Total: {style_total[3]}")
    print(f"Realism Correct: {style_currect[3]}")
    print(f"Abstract Accuracy: {style_accuract[4] * 100:.2f}%")
    print(f"Abstract Total: {style_total[4]}")
    print(f"Abstract Correct: {style_currect[4]}")

